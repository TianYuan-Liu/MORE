//! `more-rs` — Rust port of the MORE regulatory-model kernel.
//!
//! Drop-in for `PaintomicsServer/src/common/bioscripts/runMORE.R`: same CLI
//! options, same output filenames, same file formats. See `SPEC.md` for the
//! rule-by-rule mapping onto the R reference in `../R/`, which is never edited
//! and stays the oracle for the equivalence harness.

mod cli;
mod collinearity;
mod data;
mod design;
mod elasticnet;
mod jackknife;
mod matrix;
mod model;
mod output;
mod pls;
mod prep;
mod rrng;

#[cfg(test)]
mod oracle_test;

use cli::{Method, Options};
use data::Frame;
use prep::Omic;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::process::ExitCode;

/// Equivalence instrument, not part of the CLI contract: given the path to a
/// design matrix dumped by `equivalence/en_probe.R` (response in column 1),
/// print the same per-alpha columns that script prints from `cv.glmnet`, then
/// exit. Kept off the option parser so `--help` still mirrors `runMORE.R`
/// exactly.
fn en_probe(path: &str) -> ExitCode {
    // Header row of variable names, then one row per sample; response first.
    // `write.table(..., row.names = FALSE)` shape, parsed here rather than
    // through `Frame` because `Frame` is feature-by-sample with row labels.
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("MORE ERROR: cannot read {path}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut lines = text.lines();
    let ncol = match lines.next() {
        Some(h) => h.split('\t').count(),
        None => {
            eprintln!("MORE ERROR: {path} is empty");
            return ExitCode::FAILURE;
        }
    };
    let mut y = Vec::new();
    let mut cols: Vec<Vec<f64>> = vec![Vec::new(); ncol - 1];
    for line in lines.filter(|l| !l.trim().is_empty()) {
        let vals: Vec<f64> = line.split('\t').map(|v| v.trim().parse().unwrap_or(f64::NAN)).collect();
        if vals.len() != ncol {
            eprintln!("MORE ERROR: ragged row in {path}");
            return ExitCode::FAILURE;
        }
        y.push(vals[0]);
        for (j, c) in cols.iter_mut().enumerate() {
            c.push(vals[j + 1]);
        }
    }
    let x = matrix::Mat::from_columns(&cols);
    println!("   probe {} x {}", x.nrow(), x.ncol());
    if let Ok(a) = std::env::var("MORE_RS_CV_CURVE") {
        let alpha: f64 = a.parse().unwrap_or(0.1);
        let t: f64 = std::env::var("MORE_RS_EN_THRESH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(elasticnet::DESCENT_THRESH);
        for (i, (lam, cvm, cvsd, nz)) in
            elasticnet::cv_curve(&x, &y, alpha, t).iter().enumerate()
        {
            println!("   {:3} lambda={:.10e} cvm={:.10e} cvsd={:.10e} nz={}", i + 1, lam, cvm, cvsd, nz);
        }
        return ExitCode::SUCCESS;
    }
    if let Ok(a) = std::env::var("MORE_RS_EN_PATH") {
        let alpha: f64 = a.parse().unwrap_or(1.0);
        let t: f64 = std::env::var("MORE_RS_EN_THRESH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1e-5);
        let want: Option<usize> = std::env::var("MORE_RS_EN_BETA").ok().and_then(|v| v.parse().ok());
        for (i, (lam, dev, df)) in elasticnet::path_probe(&x, &y, alpha, t).iter().enumerate() {
            println!("   {:3} lambda={:.6e} dev={:.6} df={}", i + 1, lam, dev, df);
            if want == Some(i + 1) {
                for (j, b) in elasticnet::path_coefficients(&x, &y, alpha, t, i).iter().enumerate() {
                    println!("   BETA {j} {b:.12e}");
                }
            }
        }
        return ExitCode::SUCCESS;
    }
    let mut best: Option<&elasticnet::AlphaDiag> = None;
    // MORE passes `thres = epsilon` = 1e-5; the override exists so the
    // convergence tolerance can be held fixed while the path-truncation rule
    // is measured on its own.
    let thresh: f64 = std::env::var("MORE_RS_EN_THRESH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1e-5);
    let diags = elasticnet::cv_probe(&x, &y, &elasticnet::default_alphas(), thresh);
    for d in &diags {
        println!(
            "   a={:.1} nlam={:3} lmax={:.6} lmin_path={:.6} lambda.min={:.6} cvm={:.6} cvup={:.6} nz={}",
            d.alpha, d.nlam, d.lmax, d.lmin_path, d.lambda_min, d.cvm, d.cvup, d.nonzero
        );
        if best.map_or(true, |b| d.cvup < b.cvup) {
            best = Some(d);
        }
    }
    if let Some(b) = best {
        println!("   WINNER a={:.1} cvup={:.6}", b.alpha, b.cvup);
    }
    ExitCode::SUCCESS
}

fn main() -> ExitCode {
    if let Some(p) = std::env::var_os("MORE_RS_EN_PROBE") {
        return en_probe(&p.to_string_lossy());
    }
    let opts = match Options::parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("MORE ERROR: {e}");
            return ExitCode::FAILURE;
        }
    };
    match run(&opts) {
        Ok(()) => {
            println!("MORE: Analysis complete.");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("MORE ERROR: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run(opts: &Options) -> Result<(), String> {
    println!("MORE: Starting analysis...");
    let mut target = data::read_matrix(&opts.target_file)?;
    let condition = data::read_matrix(&opts.condition_file)?;
    println!("MORE: Loaded target data with {} features.", target.nrow());

    let mut raw_omics: Vec<Frame> = Vec::new();
    for path in &opts.data_files {
        raw_omics.push(data::read_matrix(path)?);
    }

    // --- sample alignment: strict, name-based, no positional fallback -----
    let samples = data::common_samples(&target, &condition, &raw_omics);
    println!("MORE: Found {} common samples among all datasets.", samples.len());
    if samples.is_empty() {
        return Err(sample_mismatch_message(&target, &condition, &raw_omics, &opts.omic_names));
    }
    target = target.select_columns(&samples)?;
    let condition = condition.select_rows(&samples);
    for f in raw_omics.iter_mut() {
        *f = f.select_columns(&samples)?;
    }

    // --- ID mangling and collision prefixing (GetPLS:120-158) -------------
    //
    // The regulator ids AS THE USER WROTE THEM, kept before anything rewrites
    // them. `read_associations` decides which column holds the regulator by
    // counting matches against this list, and the association file on disk
    // still holds the user's spelling -- so orientation has to be judged on
    // that spelling. Handing it the rewritten names instead made the match
    // count zero in both columns and aborted the job, on two ordinary inputs:
    // a regulator id that is also a target id (a TF is itself a gene), and any
    // id containing ":" (peak and region ids, which `mangle_id` rewrites to
    // "-"). R does not have the problem because runMORE.R orients the file
    // before more() rewrites anything, and MORE_PLS.R:229-234 then applies the
    // same gsub to `associations[[i]][[2]]` and to the rownames together.
    let raw_regulator_ids: Vec<Vec<String>> =
        raw_omics.iter().map(|f| f.row_names.clone()).collect();
    for f in raw_omics.iter_mut() {
        for name in f.row_names.iter_mut() {
            *name = prep::mangle_id(name);
        }
    }
    let target_ids: HashSet<&String> = target.row_names.iter().collect();
    let mut prefix_needed = vec![false; raw_omics.len()];
    for (i, f) in raw_omics.iter().enumerate() {
        if f.row_names.iter().any(|r| target_ids.contains(r)) {
            prefix_needed[i] = true;
        }
    }
    for i in 0..raw_omics.len() {
        for j in (i + 1)..raw_omics.len() {
            let a: HashSet<&String> = raw_omics[i].row_names.iter().collect();
            if raw_omics[j].row_names.iter().any(|r| a.contains(r)) {
                prefix_needed[i] = true;
                prefix_needed[j] = true;
            }
        }
    }
    for (i, f) in raw_omics.iter_mut().enumerate() {
        if prefix_needed[i] {
            let p = format!("{}-", opts.omic_names[i]);
            for name in f.row_names.iter_mut() {
                *name = format!("{p}{name}");
            }
        }
    }
    // Which omics actually carry a prefix, so the writers can take it off
    // again. The prefix is internal disambiguation: R removes it before
    // writing (output_analysis.R:414-416), and PaintOmics matches the user's
    // significant-regulator file against the half of `GENE:::REGULATOR` after
    // the separator (MOREServlet.py:1219), so a prefix left in the output
    // silently drops every red star for that omic while the job still reports
    // success.
    let omic_prefixes: HashMap<String, String> = opts
        .omic_names
        .iter()
        .enumerate()
        .filter(|(i, _)| prefix_needed[*i])
        .map(|(_, name)| (name.clone(), format!("{name}-")))
        .collect();

    // --- associations ------------------------------------------------------
    let mut assoc_per_omic: Vec<Option<Vec<data::Association>>> = Vec::new();
    for (i, path) in opts.assoc_files.iter().enumerate() {
        match path {
            None => {
                println!(
                    "MORE: No association file provided for {}. MORE will use all-to-all or internal mapping.",
                    opts.omic_names[i]
                );
                assoc_per_omic.push(None);
            }
            Some(p) => {
                let mut rows = data::read_associations(
                    p,
                    &opts.omic_names[i],
                    &raw_regulator_ids[i],
                    &target.row_names,
                )?;
                // Association regulator IDs get the same mangling and prefixing
                // as the data IDs, or the two stop matching.
                for a in rows.iter_mut() {
                    a.regulator = prep::mangle_id(&a.regulator);
                    if prefix_needed[i] {
                        a.regulator = format!("{}-{}", opts.omic_names[i], a.regulator);
                    }
                }
                assoc_per_omic.push(Some(rows));
            }
        }
    }

    // --- sample-level missing filter (applies across every matrix) --------
    let mut drop_samples: HashSet<String> = HashSet::new();
    for f in &raw_omics {
        drop_samples.extend(prep::high_na_columns(f));
    }
    if !drop_samples.is_empty() {
        let keep: Vec<String> =
            samples.iter().filter(|s| !drop_samples.contains(*s)).cloned().collect();
        println!("MORE: Number of observations with missing values: {}", drop_samples.len());
        target = target.select_columns(&keep)?;
        for f in raw_omics.iter_mut() {
            *f = f.select_columns(&keep)?;
        }
    }
    let kept_samples = target.col_names.clone();
    let condition = condition.select_rows(&kept_samples);
    let groups = prep::group_labels(&condition);
    // A design with nothing to contrast is refused here rather than modelled.
    // See prep::design_problem for why it is refused and not reinterpreted.
    if let Some(problem) = prep::design_problem(&groups) {
        return Err(problem);
    }
    // MLR drops the reference level; PLS1 keeps every level. See prep::design_columns.
    let design_cols = prep::design_columns(&groups, opts.method == Method::Mlr);
    let design_values = prep::design_matrix(&groups, &design_cols);
    // The rpc table's condition columns are ordered differently from the design
    // matrix's — see prep::rpc_columns.
    let rpc_cols = prep::rpc_columns(&groups);

    // --- per-omic regulator filters ---------------------------------------
    let mut omics: Vec<Omic> = Vec::new();
    for (i, frame) in raw_omics.into_iter().enumerate() {
        let omic_type = prep::is_binary(&frame);
        let removed_na = prep::high_na_rows(&frame);
        let surviving: Vec<String> =
            frame.row_names.iter().filter(|r| !removed_na.contains(*r)).cloned().collect();
        let after_na = frame.select_rows(&surviving);
        let removed_lv = prep::low_variation(&after_na, &groups, omic_type, opts.min_variation[i]);
        let modelled: Vec<String> =
            after_na.row_names.iter().filter(|r| !removed_lv.contains(*r)).cloned().collect();
        let data_final = after_na.select_rows(&modelled);

        println!(
            "MORE: {} — {} regulators kept, {} dropped for missing values, {} for low variation.",
            opts.omic_names[i],
            data_final.nrow(),
            removed_na.len(),
            removed_lv.len()
        );

        let mut by_target: HashMap<String, Vec<data::Association>> = HashMap::new();
        if let Some(rows) = &assoc_per_omic[i] {
            for a in rows {
                by_target.entry(a.target.clone()).or_default().push(a.clone());
            }
        }
        omics.push(Omic {
            name: opts.omic_names[i].clone(),
            data: data_final,
            input_data: frame,
            associations: assoc_per_omic[i].clone(),
            omic_type,
            removed_na,
            removed_lv,
            by_target,
        });
    }

    if omics.iter().all(|o| o.data.nrow() == 0) {
        return Err(
            "No regulators left after LowVariation filter. Consider being less restrictive.".into(),
        );
    }

    // --- target filters ----------------------------------------------------
    let any_assoc = omics.iter().any(|o| o.associations.is_some());
    let has_assoc = |t: &str| {
        omics.iter().filter(|o| o.associations.is_some()).any(|o| o.by_target.contains_key(t))
    };
    let (targets, problems) = prep::filter_targets(&target, &has_assoc, any_assoc);
    if !problems.is_empty() {
        println!("MORE: {} target features excluded before modelling.", problems.len());
    }
    println!(
        "MORE: Running model (method = {}) over {} target features...",
        opts.method,
        targets.len()
    );

    // --- fit ---------------------------------------------------------------
    let params =
        model::FitParams {
            seed: opts.seed,
            alpha: opts.alpha,
            vip: opts.vip,
            interactions: opts.interactions,
            method: opts.method,
            correlation: 0.7,
        };
    let results = model::fit_all(&targets, &target, &omics, &design_cols, &design_values, &params);

    // --- did anything actually fit? ---------------------------------------
    //
    // Every target that fails to produce a model already records why, per
    // target, honestly. What was missing is anyone reading those records:
    // `rpc_table` drops a target with no model exactly the way it drops a
    // target whose model found nothing, so a run where *no* target fitted at
    // all still printed "Analysis complete." and exited 0 over a 0-byte table.
    //
    // Reachable by uploading regulatory data with missing values in it. The
    // two `PERC_NA` filters are per regulator and per sample, so one NA per
    // regulator spread across the samples passes both; the NAs then reach the
    // design matrix and `pls::fit` returns `None` for every target. Measured
    // on a 120-target fixture: 120 x "No significant components on PLS" on the
    // PLS1 path, 120 x "No model could be fitted" on the MLR path, exit 0 in
    // both. R stops with an error on the same input.
    //
    // The test is "did any target get a model", not "did any target get a
    // significant regulator". A dataset with no real signal fits fine and
    // reports nothing, and that is a legitimate, successful answer.
    let fitted = results.iter().filter(|r| r.r2.is_some()).count();
    if !results.is_empty() {
        println!(
            "MORE: {} of {} target features produced a model.",
            fitted,
            results.len()
        );
    }
    if !results.is_empty() && fitted == 0 {
        let mut tally: std::collections::BTreeMap<&str, usize> = std::collections::BTreeMap::new();
        for r in &results {
            *tally.entry(r.problem.unwrap_or("Unknown")).or_default() += 1;
        }
        let reasons = tally
            .iter()
            .map(|(reason, n)| format!("{n} x {reason}"))
            .collect::<Vec<_>>()
            .join("; ");
        return Err(format!(
            "None of the {} target features produced a model, so there is nothing to report: \
             {reasons}. The usual cause is missing or infinite values in the regulatory data. \
             A regulator is kept whenever no more than {:.0}% of its samples are NA, and an \
             infinite value is not counted as missing at all, so either kind carries through \
             into every model from there. Check the regulatory data and either impute the \
             offending values or drop the regulators that carry them.",
            results.len(),
            prep::PERC_NA * 100.0
        ));
    }

    // --- write -------------------------------------------------------------
    let dir = Path::new(&opts.output_dir);
    std::fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;

    let rows = output::rpc_table(&results, &target, &rpc_cols, opts.filter_r2);
    output::write_rpc(
        dir,
        &opts.date_seed,
        &rows,
        &rpc_cols,
        opts.method == Method::Mlr,
        &omic_prefixes,
    )?;
    println!(
        "MORE: wrote RegulationPerCondition table ({} rows) to MORE_rpc_{}.tab",
        rows.len(),
        opts.date_seed
    );

    for omic in &omics {
        // `filter_r2` gates the yellow stars as well as the table. R derives
        // the pairs file FROM the already-filtered RegulationPerCondition
        // (runMORE.R:584-591), so a target the R2 filter drops must not
        // contribute stars either. Passing it only to `rpc_table` left the
        // pathway map starring regulators with no row behind them: at
        // --filter_r2 0.9 on one fixture R wrote 4 pairs and the port wrote
        // 217.
        let sig = output::significant_pairs(&results, &omic.name, opts.filter_r2);
        let full = output::full_pairs(omic, &sig);
        output::write_omic_files(
            dir,
            &opts.date_seed,
            &omic.name,
            &full,
            &sig,
            &omic.input_data,
            omic_prefixes.get(&omic.name).map(String::as_str),
        )?;
        println!(
            "MORE: {} — wrote {} pairs to values file ({} significant for yellow stars)",
            omic.name,
            full.len(),
            sig.len()
        );
    }

    Ok(())
}

/// Same diagnostic shape as `runMORE.R`: show the first few sample IDs from
/// every input so the mismatch is visible without reopening the files.
/// Positional alignment is intentionally not offered as a fallback — in a web
/// context nobody reads a console warning, and silently pairing
/// differently-named samples by column order publishes meaningless results.
fn sample_mismatch_message(
    target: &Frame,
    condition: &Frame,
    regulatory: &[Frame],
    names: &[String],
) -> String {
    let head = |v: &[String]| {
        let mut s = v.iter().take(3).cloned().collect::<Vec<_>>().join(", ");
        if v.len() > 3 {
            s.push_str(", ...");
        }
        s
    };
    let mut msg = String::from("No common sample names across input files.\n");
    msg.push_str(&format!("  Target samples:  {}\n", head(&target.col_names)));
    msg.push_str(&format!("  Condition rows:  {}\n", head(&condition.row_names)));
    for (i, f) in regulatory.iter().enumerate() {
        let name = names.get(i).map(|s| s.as_str()).unwrap_or("omic");
        msg.push_str(&format!("  {name} samples: {}\n", head(&f.col_names)));
    }
    msg.push_str(
        "Paintomics requires the same biological sample to carry the SAME column name in the \
         target expression file, the condition file, and every regulatory omic file. \
         Positional alignment is intentionally not used.",
    );
    msg
}
