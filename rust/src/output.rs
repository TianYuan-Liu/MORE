//! `RegulationPerCondition` and the four output files.
//!
//! Ports `RegulationPerCondition`'s PLS-with-design branch
//! (`../R/output_analysis.R:279-419`) and the writer block of `runMORE.R:501-602`.
//!
//! Filenames, headers and the `TARGET:::REGULATOR` key shape are a contract
//! with the Python side: `Job.parseGeneBasedFiles` looks a values row up in the
//! pairs file by that key, so a mismatch silently removes every significance
//! marker for the omic and shifts pathway enrichment with it.
//!
//! **Known upstream defect, deliberately not reproduced.** MORE ends
//! `RegulationPerCondition` with an unanchored global
//! `gsub("<omic>-", "", regulator)`, so a genuine regulator `TF-1` under an omic
//! named `TF` is emitted as `1`, disagreeing with the values file, which is
//! written from the input data. `runMORE.R` repairs this after the fact. This
//! port emits the true IDs directly; the equivalence harness compares against
//! the repaired R output, not the raw table.

use crate::data::Frame;
use crate::model::{comparable_beta, TargetResult};
use crate::prep::Filter;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::Path;

/// One row of the RegulationPerCondition table.
pub struct RpcRow {
    pub target: String,
    pub regulator: String,
    pub omic: String,
    pub area: String,
    /// One coefficient per `Group_*` column, in `design_cols` order.
    pub betas: Vec<f64>,
    pub r2: Option<f64>,
    /// Real regulator ID of the collinearity-group representative this row
    /// stands with; empty for a regulator that is in no group.
    pub representative: String,
}

/// Build the rpc table for one target.
///
/// The assignment rules come straight from the R, and the distinction between
/// them matters: a regulator appearing *only* on its own has its coefficient
/// **set** on every condition, whereas one that also appears in interactions
/// **accumulates** — its main effect into every condition and each interaction
/// term into its own condition.
fn rpc_rows_for(result: &TargetResult, design_cols: &[String], target_sd: f64) -> Vec<RpcRow> {
    if result.significant.is_empty() {
        return Vec::new();
    }
    let n_groups = design_cols.len().max(1);

    let model_variables: Vec<&str> =
        result.coefficients.iter().map(|(v, _, _)| v.as_str()).collect();
    let interaction_vars: Vec<&str> =
        model_variables.iter().copied().filter(|v| v.contains(':')).collect();
    // The regulator side of each interaction — R takes every second element of
    // the flattened split, i.e. the part after the colon.
    let inter_regulators: Vec<&str> = interaction_vars
        .iter()
        .filter_map(|v| v.split(':').nth(1))
        .collect();

    let plain_only: Vec<&str> = model_variables
        .iter()
        .copied()
        .filter(|v| !interaction_vars.contains(v))
        .filter(|v| !inter_regulators.contains(v))
        .filter(|v| !v.starts_with("Group_"))
        .collect();
    let inter_and_plain: Vec<&str> = inter_regulators
        .iter()
        .copied()
        .filter(|v| model_variables.contains(v))
        .collect();
    let inter_only: Vec<&str> = inter_regulators
        .iter()
        .copied()
        .filter(|v| !model_variables.contains(v))
        .collect();

    // One row per significant regulator, carrying its omic and area.
    let mut rows: Vec<RpcRow> = Vec::new();
    let mut where_is: HashMap<&str, usize> = HashMap::new();
    for reg in &result.significant {
        let Some(meta) = result.regulators.iter().find(|r| r.regulator == *reg) else {
            continue;
        };
        where_is.insert(reg.as_str(), rows.len());
        rows.push(RpcRow {
            target: result.target.clone(),
            regulator: reg.clone(),
            omic: meta.omic.clone(),
            area: meta.area.clone(),
            betas: vec![0.0; n_groups],
            r2: result.r2,
            representative: String::new(),
        });
    }

    for (variable, beta, _) in &result.coefficients {
        let parts: Vec<&str> = variable.split(':').collect();
        let group_part = parts.iter().copied().find(|p| p.starts_with("Group_"));
        let reg_part: Vec<&str> =
            parts.iter().copied().filter(|p| !p.starts_with("Group_")).collect();

        let significant = parts.iter().any(|p| result.significant.iter().any(|s| s == p));
        if !significant {
            continue;
        }

        if parts.iter().any(|p| plain_only.contains(p)) {
            if let Some(&r) = reg_part.first().and_then(|p| where_is.get(p)) {
                for b in rows[r].betas.iter_mut() {
                    *b = *beta;
                }
            }
        }
        if parts.iter().any(|p| inter_only.contains(p)) {
            if let (Some(&r), Some(g)) = (reg_part.first().and_then(|p| where_is.get(p)), group_part)
            {
                if let Some(gi) = design_cols.iter().position(|c| c == g) {
                    rows[r].betas[gi] += beta;
                }
            }
        }
        if parts.iter().any(|p| inter_and_plain.contains(p)) {
            match group_part {
                None => {
                    if let Some(&r) = reg_part.first().and_then(|p| where_is.get(p)) {
                        for b in rows[r].betas.iter_mut() {
                            *b += beta;
                        }
                    }
                }
                Some(g) => {
                    if let Some(&r) = reg_part.first().and_then(|p| where_is.get(p)) {
                        if let Some(gi) = design_cols.iter().position(|c| c == g) {
                            rows[r].betas[gi] += beta;
                        }
                    }
                }
            }
        }
    }

    // A collapsed clique has exactly one column in the design -- the
    // representative's -- so only its row picked up a coefficient above. R
    // hands that coefficient to every member, negated for members that
    // correlate negatively with the representative (the `_N` marker), and
    // records the representative's real ID alongside
    // (`output_analysis.R:300-335`). Without this pass every non-representative
    // member reports a beta of zero while still counting as a reported edge.
    for group in &result.groups {
        let Some(&rep_row) = where_is.get(group.representative.as_str()) else {
            continue;
        };
        let rep_betas = rows[rep_row].betas.clone();
        for (member, sign) in group.members.iter().zip(&group.signs) {
            let Some(&mi) = where_is.get(member.as_str()) else {
                continue;
            };
            rows[mi].representative = group.representative.clone();
            if mi != rep_row {
                rows[mi].betas = rep_betas.iter().map(|b| b * sign).collect();
            }
        }
    }

    for row in rows.iter_mut() {
        for b in row.betas.iter_mut() {
            *b = comparable_beta(*b, target_sd);
        }
    }
    rows
}

/// Build the whole rpc table, honouring `--filter_r2`.
pub fn rpc_table(
    results: &[TargetResult],
    target_data: &Frame,
    design_cols: &[String],
    filter_r2: f64,
) -> Vec<RpcRow> {
    let index = target_data.row_index();
    let mut out = Vec::new();
    for result in results {
        if let Some(r2) = result.r2 {
            if !(r2 > filter_r2) {
                continue;
            }
        } else {
            continue;
        }
        let sd = index
            .get(result.target.as_str())
            .map(|&r| crate::matrix::sd(&target_data.values[r]))
            .unwrap_or(1.0);
        out.extend(rpc_rows_for(result, design_cols, sd));
    }
    out
}

/// R's `as.character(double)`, which `write.table` uses: `%.15g` with trailing
/// zeros trimmed. Values reaching the rpc table are already `signif(x, 4)`, so
/// this is short in practice; the values file carries raw input numbers, where
/// the formatting actually has to hold up.
pub fn format_r_double(v: f64) -> String {
    if v.is_nan() {
        return "NaN".into();
    }
    if v.is_infinite() {
        return if v > 0.0 { "Inf".into() } else { "-Inf".into() };
    }
    if v == 0.0 {
        return "0".into();
    }
    const P: i32 = 15;
    let e = v.abs().log10().floor() as i32;
    let mut s = if e < -5 || e >= P {
        let mantissa_digits = (P - 1).max(0) as usize;
        let formatted = format!("{:.*e}", mantissa_digits, v);
        // Rust writes `1.5e-7`; R writes `1.5e-07`.
        match formatted.split_once('e') {
            Some((m, exp)) => {
                let m = trim_zeros(m);
                let (sign, digits) = if let Some(d) = exp.strip_prefix('-') {
                    ("-", d)
                } else {
                    ("+", exp.trim_start_matches('+'))
                };
                format!("{m}e{sign}{:0>2}", digits)
            }
            None => formatted,
        }
    } else {
        let decimals = (P - 1 - e).max(0) as usize;
        trim_zeros(&format!("{:.*}", decimals, v))
    };
    if s == "-0" {
        s = "0".into();
    }
    s
}

fn trim_zeros(s: &str) -> String {
    if !s.contains('.') {
        return s.to_string();
    }
    let t = s.trim_end_matches('0');
    t.trim_end_matches('.').to_string()
}

fn write_lines(path: &Path, lines: &[String]) -> Result<(), String> {
    let mut f = fs::File::create(path).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    for line in lines {
        writeln!(f, "{line}").map_err(|e| format!("cannot write {}: {e}", path.display()))?;
    }
    Ok(())
}

/// Write `MORE_rpc_<seed>.tab`. Always created, even with zero rows — an absent
/// file means MORE never ran, which the Python side distinguishes.
pub fn write_rpc(
    dir: &Path,
    seed: &str,
    rows: &[RpcRow],
    design_cols: &[String],
    // The MLR branch of RegulationPerCondition keeps the collinearity-group
    // "representative" column; the PLS branch drops it with myresults[, -5]
    // (output_analysis.R:410). Same table, different width by method.
    representative: bool,
) -> Result<(), String> {
    let path = dir.join(format!("MORE_rpc_{seed}.tab"));
    if rows.is_empty() {
        return write_lines(&path, &[]);
    }
    let mut header = vec![
        "targetF".to_string(),
        "regulator".to_string(),
        "omic".to_string(),
        "area".to_string(),
    ];
    if representative {
        header.push("representative".to_string());
    }
    header.extend(design_cols.iter().cloned());
    header.push("R2".to_string());

    let mut lines = vec![header.join("\t")];
    for row in rows {
        let mut fields = vec![
            row.target.clone(),
            row.regulator.clone(),
            row.omic.clone(),
            row.area.clone(),
        ];
        if representative {
            // R names the representative's real regulator ID here and leaves
            // it blank for a regulator in no group (filter == "Model").
            fields.push(row.representative.clone());
        }
        fields.extend(row.betas.iter().map(|b| format_r_double(*b)));
        // na = "" in write.table.
        fields.push(row.r2.map(format_r_double).unwrap_or_default());
        lines.push(fields.join("\t"));
    }
    write_lines(&path, &lines)
}

/// The per-omic trio. `pairs` is significance-filtered; `assoc` and the values
/// file carry **every** input pair whose regulator exists in the data matrix,
/// because PA Step 1 needs the unfiltered snapshot.
pub fn write_omic_files(
    dir: &Path,
    seed: &str,
    omic: &str,
    full_pairs: &[(String, String)],
    significant_pairs: &[(String, String)],
    reg_data: &Frame,
) -> Result<(), String> {
    let assoc_path = dir.join(format!("MORE_relevant_assoc_{omic}_{seed}.tab"));
    let assoc: Vec<String> =
        full_pairs.iter().map(|(t, r)| format!("{t}\t{r}")).collect();
    write_lines(&assoc_path, &assoc)?;

    let pairs_path = dir.join(format!("MORE_relevant_pairs_{omic}_{seed}.tab"));
    let mut seen = Vec::new();
    for (t, r) in significant_pairs {
        let key = format!("{t}:::{r}");
        if !seen.contains(&key) {
            seen.push(key);
        }
    }
    write_lines(&pairs_path, &seen)?;

    // Values file: header is "# Gene name" then the sample names, and R's NA is
    // written as the literal NaN because PA Step 1 calls float() on every
    // value and float("NA") raises.
    let values_path = dir.join(format!("MORE_output_{omic}_{seed}.tab"));
    let index = reg_data.row_index();
    let mut lines = vec![format!("# Gene name\t{}", reg_data.col_names.join("\t"))];
    for (t, r) in full_pairs {
        let Some(&row) = index.get(r.as_str()) else {
            continue;
        };
        let vals: Vec<String> =
            reg_data.values[row].iter().map(|v| format_r_double(*v)).collect();
        lines.push(format!("{t}:::{r}\t{}", vals.join("\t")));
    }
    write_lines(&values_path, &lines)
}

/// Every input pair whose regulator survives into the regulator matrix.
pub fn full_pairs(omic: &crate::prep::Omic) -> Vec<(String, String)> {
    let mut out: Vec<(String, String)> = Vec::new();
    match &omic.associations {
        Some(rows) => {
            for a in rows {
                if omic.data.row_names.contains(&a.regulator) {
                    let pair = (a.target.clone(), a.regulator.clone());
                    if !out.contains(&pair) {
                        out.push(pair);
                    }
                }
            }
        }
        None => {}
    }
    out
}

/// Significant (target, regulator) pairs for one omic.
pub fn significant_pairs(results: &[TargetResult], omic: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    for r in results {
        for reg in &r.significant {
            let belongs = r
                .regulators
                .iter()
                .any(|m| m.regulator == *reg && m.omic == omic && m.filter == Filter::Model);
            if belongs {
                out.push((r.target.clone(), reg.clone()));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::design::RegulatorRow;

    fn reg(name: &str, omic: &str) -> RegulatorRow {
        RegulatorRow {
            regulator: name.into(),
            omic: omic.into(),
            area: String::new(),
            filter: Filter::Model,
        }
    }

    fn result_with(coeffs: Vec<(&str, f64)>, significant: Vec<&str>) -> TargetResult {
        TargetResult {
            target: "G1".into(),
            regulators: vec![reg("R1", "TF"), reg("R2", "TF")],
            significant: significant.into_iter().map(|s| s.to_string()).collect(),
            coefficients: coeffs.into_iter().map(|(v, b)| (v.to_string(), b, 0.01)).collect(),
            groups: Vec::new(),
            r2: Some(0.9),
            q2: Some(0.8),
            rmsee: Some(0.1),
            ncomp: Some(1),
            problem: None,
        }
    }

    fn groups() -> Vec<String> {
        vec!["Group_A".to_string(), "Group_B".to_string()]
    }

    #[test]
    fn a_plain_only_regulator_sets_the_same_beta_on_every_condition() {
        let r = result_with(vec![("R1", 2.0)], vec!["R1"]);
        let rows = rpc_rows_for(&r, &groups(), 1.0);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].betas, vec![2.0, 2.0]);
    }

    #[test]
    fn interaction_terms_give_condition_specific_betas() {
        let r = result_with(vec![("Group_A:R1", 3.0), ("Group_B:R1", -1.0)], vec!["R1"]);
        let rows = rpc_rows_for(&r, &groups(), 1.0);
        assert_eq!(rows[0].betas, vec![3.0, -1.0]);
    }

    #[test]
    fn a_main_effect_plus_an_interaction_accumulates() {
        let r = result_with(vec![("R1", 1.0), ("Group_A:R1", 0.5)], vec!["R1"]);
        let rows = rpc_rows_for(&r, &groups(), 1.0);
        // main effect into both conditions, interaction into Group_A only
        assert_eq!(rows[0].betas, vec![1.5, 1.0]);
    }

    #[test]
    fn betas_are_divided_by_the_target_sd() {
        let r = result_with(vec![("R1", 2.0)], vec!["R1"]);
        let rows = rpc_rows_for(&r, &groups(), 4.0);
        assert_eq!(rows[0].betas, vec![0.5, 0.5]);
    }

    /// A clique whose representative is R1 and whose other member correlates
    /// with the sign given.
    fn clique(sign: f64) -> Vec<crate::collinearity::Group> {
        vec![crate::collinearity::Group {
            name: "TF_mc1_R".into(),
            representative: "R1".into(),
            members: vec!["R1".into(), "R2".into()],
            signs: vec![1.0, sign],
        }]
    }

    #[test]
    fn a_clique_member_inherits_the_representatives_beta() {
        // Only R1 has a design column; R2 rides along through the group.
        let mut r = result_with(vec![("R1", 2.0)], vec!["R1", "R2"]);
        r.groups = clique(1.0);
        let rows = rpc_rows_for(&r, &groups(), 1.0);
        let r2 = rows.iter().find(|x| x.regulator == "R2").expect("R2 row");
        assert_eq!(r2.betas, vec![2.0, 2.0]);
    }

    #[test]
    fn a_negatively_correlated_member_inherits_the_opposite_sign() {
        let mut r = result_with(vec![("R1", 2.0)], vec!["R1", "R2"]);
        r.groups = clique(-1.0);
        let rows = rpc_rows_for(&r, &groups(), 1.0);
        let r2 = rows.iter().find(|x| x.regulator == "R2").expect("R2 row");
        assert_eq!(r2.betas, vec![-2.0, -2.0]);
    }

    #[test]
    fn every_clique_member_names_the_representative() {
        let mut r = result_with(vec![("R1", 2.0)], vec!["R1", "R2"]);
        r.groups = clique(1.0);
        let rows = rpc_rows_for(&r, &groups(), 1.0);
        for row in &rows {
            assert_eq!(row.representative, "R1", "{}", row.regulator);
        }
    }

    #[test]
    fn a_regulator_in_no_clique_leaves_the_representative_column_empty() {
        let r = result_with(vec![("R1", 2.0)], vec!["R1"]);
        let rows = rpc_rows_for(&r, &groups(), 1.0);
        assert_eq!(rows[0].representative, "");
    }

    #[test]
    fn the_representative_keeps_its_own_beta_unscaled_by_its_sign() {
        let mut r = result_with(vec![("R1", 2.0)], vec!["R1", "R2"]);
        r.groups = clique(-1.0);
        let rows = rpc_rows_for(&r, &groups(), 1.0);
        let r1 = rows.iter().find(|x| x.regulator == "R1").expect("R1 row");
        assert_eq!(r1.betas, vec![2.0, 2.0]);
    }

    #[test]
    fn a_target_with_no_significant_regulators_contributes_no_rows() {
        let r = result_with(vec![], vec![]);
        assert!(rpc_rows_for(&r, &groups(), 1.0).is_empty());
    }

    #[test]
    fn r_double_formatting_matches_common_cases() {
        assert_eq!(format_r_double(0.5), "0.5");
        assert_eq!(format_r_double(2.0), "2");
        assert_eq!(format_r_double(-0.1235), "-0.1235");
        assert_eq!(format_r_double(0.0), "0");
        assert_eq!(format_r_double(1.0 / 3.0), "0.333333333333333");
    }

    #[test]
    fn r_double_formatting_pads_the_exponent_like_r() {
        assert_eq!(format_r_double(1.5e-7), "1.5e-07");
    }

    #[test]
    fn missing_values_are_written_as_nan_for_the_python_validator() {
        assert_eq!(format_r_double(f64::NAN), "NaN");
    }

    #[test]
    fn the_rpc_file_is_created_even_with_no_rows() {
        let dir = std::env::temp_dir().join(format!("more_rs_rpc_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        write_rpc(&dir, "seed", &[], &groups(), false).unwrap();
        let p = dir.join("MORE_rpc_seed.tab");
        assert!(p.exists());
        assert_eq!(fs::read_to_string(&p).unwrap(), "");
    }

    #[test]
    fn the_rpc_header_carries_one_column_per_condition_plus_r2() {
        let dir = std::env::temp_dir().join(format!("more_rs_rpc2_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let rows = vec![RpcRow {
            target: "G1".into(),
            regulator: "R1".into(),
            omic: "TF".into(),
            area: String::new(),
            representative: String::new(),
            betas: vec![1.0, 2.0],
            r2: Some(0.9),
        }];
        write_rpc(&dir, "seed", &rows, &groups(), false).unwrap();
        let text = fs::read_to_string(dir.join("MORE_rpc_seed.tab")).unwrap();
        let mut lines = text.lines();
        assert_eq!(lines.next().unwrap(), "targetF\tregulator\tomic\tarea\tGroup_A\tGroup_B\tR2");
        assert_eq!(lines.next().unwrap(), "G1\tR1\tTF\t\t1\t2\t0.9");
    }

    #[test]
    fn the_pairs_file_uses_the_triple_colon_key() {
        let dir = std::env::temp_dir().join(format!("more_rs_pairs_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let data = Frame {
            row_names: vec!["R1".into()],
            col_names: vec!["S1".into(), "S2".into()],
            values: vec![vec![1.0, f64::NAN]],
        };
        write_omic_files(
            &dir,
            "seed",
            "TF",
            &[("G1".to_string(), "R1".to_string())],
            &[("G1".to_string(), "R1".to_string())],
            &data,
        )
        .unwrap();
        let pairs = fs::read_to_string(dir.join("MORE_relevant_pairs_TF_seed.tab")).unwrap();
        assert_eq!(pairs.trim(), "G1:::R1");
        let assoc = fs::read_to_string(dir.join("MORE_relevant_assoc_TF_seed.tab")).unwrap();
        assert_eq!(assoc.trim(), "G1\tR1");
        let values = fs::read_to_string(dir.join("MORE_output_TF_seed.tab")).unwrap();
        let mut lines = values.lines();
        assert_eq!(lines.next().unwrap(), "# Gene name\tS1\tS2");
        assert_eq!(lines.next().unwrap(), "G1:::R1\t1\tNaN");
    }
}
