//! Per-target model driver, fanned out across targets with Rayon.
//!
//! Ports `ResultsPerTargetF.i` (`../R/MORE_PLS.R:704`).
//!
//! Parallelism is across targets and never inside a fit. R's `parallel = TRUE`
//! is 3.3x *slower* on this workload because each target is only ~0.3 s of work
//! and `furrr` serialises the regulator matrices to every worker; here the
//! omics are shared by reference, so the fan-out actually pays. Any BLAS would
//! be pinned to one thread for the same reason — there is none, by design.

use crate::design::{self, Design, RegulatorRow};
use crate::jackknife;
use crate::matrix::{scale_in_place, signif, Mat};
use crate::pls;
use crate::prep::Omic;
use rayon::prelude::*;

/// Everything the output stage needs from one target.
pub struct TargetResult {
    pub target: String,
    pub regulators: Vec<RegulatorRow>,
    /// Regulator IDs judged significant, in the order R recovers them.
    pub significant: Vec<String>,
    /// (variable name, coefficient, p-value) for the significant variables
    /// only — taken from the *original* fit, as R does.
    pub coefficients: Vec<(String, f64, f64)>,
    /// `R2Y(cum)` of the reported model, already `signif(x, 3)`.
    pub r2: Option<f64>,
    pub q2: Option<f64>,
    pub rmsee: Option<f64>,
    pub ncomp: Option<usize>,
    pub problem: Option<&'static str>,
    /// Collinearity groups collapsed for this target. Empty on the PLS1 path,
    /// which does not collapse. The rpc table needs them after the fit: every
    /// member inherits the representative's coefficient, sign-flipped when the
    /// member correlates negatively with it, and names it in the
    /// `representative` column.
    pub groups: Vec<crate::collinearity::Group>,
}

impl TargetResult {
    fn failed(target: &str, regulators: Vec<RegulatorRow>, problem: &'static str) -> Self {
        TargetResult {
            target: target.to_string(),
            regulators,
            significant: Vec::new(),
            coefficients: Vec::new(),
            r2: None,
            q2: None,
            rmsee: None,
            ncomp: None,
            problem: Some(problem),
            groups: Vec::new(),
        }
    }
}

pub struct FitParams {
    pub alpha: f64,
    pub vip: f64,
    pub interactions: bool,
    pub method: crate::cli::Method,
    /// MORE's `correlation` default; the collinearity threshold on the MLR path.
    pub correlation: f64,
}

/// Fit every target. The only shared mutable state is none — each target
/// produces an independent result, which is what makes the fan-out safe.
pub fn fit_all(
    targets: &[String],
    target_data: &crate::data::Frame,
    omics: &[Omic],
    design_cols: &[String],
    design_values: &[Vec<f64>],
    params: &FitParams,
) -> Vec<TargetResult> {
    let index = target_data.row_index();
    targets
        .par_iter()
        .map(|t| {
            let row = index.get(t.as_str()).map(|&r| target_data.values[r].as_slice());
            match row {
                Some(y) => fit_one(t, y, omics, design_cols, design_values, params),
                None => TargetResult::failed(t, Vec::new(), "Target feature had no initial regulators"),
            }
        })
        .collect()
}

fn fit_one(
    target: &str,
    y_raw: &[f64],
    omics: &[Omic],
    design_cols: &[String],
    design_values: &[Vec<f64>],
    params: &FitParams,
) -> TargetResult {
    let mut regulators = design::all_regulators(target, omics);
    if regulators.is_empty() {
        return TargetResult::failed(target, regulators, "Target feature had no initial regulators");
    }
    design::classify(&mut regulators, omics);

    let n = y_raw.len();

    // MLR collapses cliques of correlated regulators BEFORE the design is
    // built, then expands the survivor back to its whole group after
    // selection. Without this the elastic net reports one edge where R
    // reports the entire clique -- the whole MLR recall gap.
    let mut groups = Vec::new();
    let mut design_rows = regulators.clone();
    if params.method == crate::cli::Method::Mlr {
        let (g, _skipped_binary) =
            crate::collinearity::find_groups(&regulators, omics, params.correlation);
        let drop = crate::collinearity::suppressed(&g);
        design_rows.retain(|r| !drop.contains(&r.regulator));
        groups = g;
    }

    let built = design::build(
        &design_rows,
        omics,
        design_cols,
        design_values,
        n,
        params.interactions,
    );
    let Some((columns, x)) = built else {
        return TargetResult::failed(target, regulators, "No regulators left after NA/LowVar filtering");
    };
    let design = Design { columns, x, regulators };

    // MLR models the response on its own scale and fits an intercept; PLS1
    // is handed a scaled response (ResultsPerTargetF.i:107).
    if params.method == crate::cli::Method::Mlr {
        // MORE_RS_DEBUG_MLR dumps the port's side of what
        // equivalence/mlr_internals_probe.R dumps for R, so the two can be
        // diffed directly instead of hypothesised about.
        if std::env::var_os("MORE_RS_DEBUG_MLR").is_some() {
            eprintln!(
                "DBG {} model_regs={} groups={} cols={}",
                target,
                design.regulators.iter().filter(|r| r.filter == crate::prep::Filter::Model).count(),
                groups.len(),
                design.columns.len()
            );
            for g in &groups {
                eprintln!("DBG   group {} members={:?}", g.name, g.members);
            }
            eprintln!("DBG   columns {:?}", design.columns);
        }
        // MORE_RS_DEBUG_DESIGN=<dir> writes this target's design matrix in the
        // same shape equivalence/design_probe.R dumps from R, response first.
        if let Some(dir) = std::env::var_os("MORE_RS_DEBUG_DESIGN") {
            let dir = std::path::PathBuf::from(dir);
            let _ = std::fs::create_dir_all(&dir);
            let mut out = String::new();
            out.push_str("response");
            for c in &design.columns {
                out.push('\t');
                out.push_str(c);
            }
            out.push('\n');
            for i in 0..n {
                out.push_str(&format!("{}", y_raw[i]));
                for j in 0..design.columns.len() {
                    out.push_str(&format!("\t{}", design.x.get(i, j)));
                }
                out.push('\n');
            }
            let _ = std::fs::write(dir.join(format!("{target}.tsv")), out);
        }
        return fit_one_mlr(target, y_raw, design, &groups);
    }

    let mut y = y_raw.to_vec();
    scale_in_place(&mut y);

    // `cross = 7`, or `nrow - 2` when there are fewer than 7 observations.
    let cross = if n < 7 { n.saturating_sub(2).max(1) } else { 7 };

    // Autofit, then MORE's retry with a single component forced.
    let fit = pls::fit(&design.x, &y, None, cross)
        .or_else(|| pls::fit(&design.x, &y, Some(1), cross));
    let Some(fit) = fit else {
        return TargetResult::failed(target, design.regulators, "No significant components on PLS");
    };

    let pvals = jackknife::p_values(&design.x, &y, &fit);

    // VIP > threshold AND jackknife p < alpha.
    let sig_idx: Vec<usize> = (0..design.columns.len())
        .filter(|&j| fit.vip[j] > params.vip && pvals[j] < params.alpha)
        .collect();

    // Reported coefficients come from the ORIGINAL fit, restricted to the
    // significant variables. The refit below only supplies goodness-of-fit —
    // getting this backwards silently changes every number in the rpc table.
    let coefficients: Vec<(String, f64, f64)> = sig_idx
        .iter()
        .map(|&j| (design.columns[j].clone(), fit.coefficients[j], pvals[j]))
        .collect();

    let reported = if sig_idx.is_empty() {
        fit
    } else {
        let sub = design.x.select_cols(&sig_idx);
        pls::fit(&sub, &y, None, cross)
            .or_else(|| pls::fit(&sub, &y, Some(1), cross))
            .unwrap_or(fit)
    };

    // Significant *regulators*, recovered from significant *variables*: an
    // interaction term Group_Treat:TF-1 credits TF-1.
    let mut significant: Vec<String> = Vec::new();
    for &j in &sig_idx {
        for reg in design::regulators_of(&design.columns[j], &design.regulators) {
            if !significant.contains(&reg) {
                significant.push(reg);
            }
        }
    }

    let problem = if significant.is_empty() {
        Some("No significant regulators after variable selection")
    } else {
        None
    };

    TargetResult {
        target: target.to_string(),
        regulators: design.regulators,
        significant,
        coefficients,
        r2: Some(reported.r2y_cum),
        q2: Some(reported.q2_cum),
        rmsee: Some(reported.rmsee),
        ncomp: Some(reported.n_comp),
        problem,
        // PLS1 does not collapse collinear regulators.
        groups: Vec::new(),
    }
}

/// MLR + elastic net, ports `ResultsPerTargetF.i.mlr` -> `ElasticNet`.
///
/// Variable selection is "coefficient survived the penalty", not a p-value
/// test, so `alpha`/`vip` play no part here. The chosen (alpha, lambda) pair
/// comes from cross-validation; see `elasticnet` for why the folds are
/// deterministic and what that costs, measured.
fn fit_one_mlr(
    target: &str,
    y_raw: &[f64],
    design: Design,
    groups: &[crate::collinearity::Group],
) -> TargetResult {
    let fit = match crate::elasticnet::cv_fit(
        &design.x,
        y_raw,
        &crate::elasticnet::default_alphas(),
        1e-5, // MORE's `epsilon`, passed to glmnet as `thres`
    ) {
        Some(f) => f,
        None => {
            return TargetResult::failed(target, design.regulators, "No model could be fitted");
        }
    };

    if std::env::var_os("MORE_RS_DEBUG_MLR").is_some() {
        eprintln!(
            "DBG {target} chose alpha={:.1} lambda={:.6} nz={} dev={:.6}",
            fit.alpha,
            fit.lambda,
            fit.coefficients.iter().filter(|b| **b != 0.0).count(),
            fit.dev_ratio
        );
    }
    let coefficients: Vec<(String, f64, f64)> = design
        .columns
        .iter()
        .zip(&fit.coefficients)
        .filter(|(_, b)| **b != 0.0)
        .map(|(name, b)| (name.clone(), *b, f64::NAN))
        .collect();

    let mut selected: Vec<String> = Vec::new();
    for (name, _, _) in &coefficients {
        for reg in design::regulators_of(name, &design.regulators) {
            if !selected.contains(&reg) {
                selected.push(reg);
            }
        }
    }
    // A selected representative stands for every member of its clique.
    let significant = crate::collinearity::expand(&selected, groups);

    let problem = if significant.is_empty() {
        Some("No significant regulators after variable selection")
    } else {
        None
    };

    TargetResult {
        target: target.to_string(),
        regulators: design.regulators,
        significant,
        coefficients,
        // modelcharac reports dev.ratio rounded to 6 digits as R.squared.
        r2: Some((fit.dev_ratio * 1e6).round() / 1e6),
        q2: None,
        rmsee: None,
        ncomp: None,
        problem,
        groups: groups.to_vec(),
    }
}

/// `ComparableBetas`: per-condition coefficients are divided by the target's
/// own standard deviation so betas are comparable across targets, then rounded
/// to four significant digits (`RegulationPerCondition:408-409`).
pub fn comparable_beta(beta: f64, target_sd: f64) -> f64 {
    signif(beta / target_sd, 4)
}

/// Unused-import guard: `Mat` is part of the public shape of a `Design`.
#[allow(dead_code)]
fn _shape(_: &Mat) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::{Association, Frame};
    use crate::prep::Filter;
    use std::collections::{HashMap, HashSet};

    fn make_omic(name: &str, regs: &[(&str, Vec<f64>)], assoc: Option<Vec<Association>>) -> Omic {
        let data = Frame {
            row_names: regs.iter().map(|(n, _)| n.to_string()).collect(),
            col_names: (0..regs[0].1.len()).map(|i| format!("S{i}")).collect(),
            values: regs.iter().map(|(_, v)| v.clone()).collect(),
        };
        let mut by_target: HashMap<String, Vec<Association>> = HashMap::new();
        if let Some(a) = &assoc {
            for row in a {
                by_target.entry(row.target.clone()).or_default().push(row.clone());
            }
        }
        Omic {
            name: name.into(),
            data,
            associations: assoc,
            omic_type: 0,
            removed_na: HashSet::new(),
            removed_lv: HashSet::new(),
            by_target,
        }
    }

    /// 12 samples, two conditions, one regulator that drives the target and one
    /// that does not.
    fn scenario() -> (Frame, Vec<Omic>, Vec<String>, Vec<Vec<f64>>) {
        let n = 12;
        let driver: Vec<f64> = (0..n).map(|i| ((i * 7 % 11) as f64) / 11.0 - 0.5).collect();
        let noise: Vec<f64> = (0..n).map(|i| ((i * 5 % 13) as f64) / 13.0 - 0.5).collect();
        let y: Vec<f64> = driver.iter().map(|v| 3.0 * v).collect();

        let target = Frame {
            row_names: vec!["G1".into()],
            col_names: (0..n).map(|i| format!("S{i}")).collect(),
            values: vec![y],
        };
        let omics = vec![make_omic(
            "TF",
            &[("DRV", driver), ("NOI", noise)],
            Some(vec![
                Association { target: "G1".into(), regulator: "DRV".into(), area: String::new() },
                Association { target: "G1".into(), regulator: "NOI".into(), area: String::new() },
            ]),
        )];
        let groups: Vec<String> =
            (0..n).map(|i| if i < 6 { "A".to_string() } else { "B".to_string() }).collect();
        let cols = crate::prep::design_columns(&groups, false);
        let values = crate::prep::design_matrix(&groups, &cols);
        (target, omics, cols, values)
    }

    #[test]
    fn the_driving_regulator_is_selected() {
        let (target, omics, cols, values) = scenario();
        let params = FitParams { alpha: 0.05, vip: 0.8, interactions: true, method: crate::cli::Method::Pls1, correlation: 0.7 };
        let res = fit_all(&["G1".to_string()], &target, &omics, &cols, &values, &params);
        assert_eq!(res.len(), 1);
        assert!(res[0].significant.contains(&"DRV".to_string()), "{:?}", res[0].significant);
    }

    #[test]
    fn every_regulator_is_reported_even_when_not_significant() {
        let (target, omics, cols, values) = scenario();
        let params = FitParams { alpha: 0.05, vip: 0.8, interactions: true, method: crate::cli::Method::Pls1, correlation: 0.7 };
        let res = fit_all(&["G1".to_string()], &target, &omics, &cols, &values, &params);
        assert_eq!(res[0].regulators.len(), 2);
        assert!(res[0].regulators.iter().all(|r| r.filter == Filter::Model));
    }

    #[test]
    fn a_target_absent_from_the_expression_matrix_is_reported_not_dropped() {
        let (target, omics, cols, values) = scenario();
        let params = FitParams { alpha: 0.05, vip: 0.8, interactions: true, method: crate::cli::Method::Pls1, correlation: 0.7 };
        let res = fit_all(&["MISSING".to_string()], &target, &omics, &cols, &values, &params);
        assert_eq!(res.len(), 1);
        assert!(res[0].problem.is_some());
    }

    #[test]
    fn goodness_of_fit_is_reported_when_a_model_exists() {
        let (target, omics, cols, values) = scenario();
        let params = FitParams { alpha: 0.05, vip: 0.8, interactions: true, method: crate::cli::Method::Pls1, correlation: 0.7 };
        let res = fit_all(&["G1".to_string()], &target, &omics, &cols, &values, &params);
        assert!(res[0].r2.is_some());
        assert!(res[0].ncomp.unwrap() >= 1);
    }

    #[test]
    fn coefficients_are_returned_only_for_significant_variables() {
        let (target, omics, cols, values) = scenario();
        let params = FitParams { alpha: 0.05, vip: 0.8, interactions: true, method: crate::cli::Method::Pls1, correlation: 0.7 };
        let res = fit_all(&["G1".to_string()], &target, &omics, &cols, &values, &params);
        assert!(!res[0].coefficients.is_empty());
        for (_, _, p) in &res[0].coefficients {
            assert!(*p < 0.05, "a non-significant variable was reported");
        }
    }

    #[test]
    fn fanning_out_gives_the_same_answer_as_one_target_at_a_time() {
        // The parallel path must not depend on how targets are batched.
        let (target, omics, cols, values) = scenario();
        let params = FitParams { alpha: 0.05, vip: 0.8, interactions: true, method: crate::cli::Method::Pls1, correlation: 0.7 };
        let many = fit_all(
            &["G1".to_string(), "G1".to_string(), "G1".to_string()],
            &target, &omics, &cols, &values, &params,
        );
        let one = fit_all(&["G1".to_string()], &target, &omics, &cols, &values, &params);
        for r in &many {
            assert_eq!(r.significant, one[0].significant);
            assert_eq!(r.r2, one[0].r2);
        }
    }

    #[test]
    fn comparable_betas_divide_by_the_target_sd_and_round_to_four_digits() {
        assert_eq!(comparable_beta(1.0, 2.0), 0.5);
        assert_eq!(comparable_beta(0.123456789, 1.0), 0.1235);
    }
}
