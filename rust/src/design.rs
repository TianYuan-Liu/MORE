//! Per-target design matrix assembly.
//!
//! Ports `GetAllReg` -> `RemovedRegulators` -> `RegulatorsInteractions` ->
//! `Scaling.type` -> the constant-column filter, i.e. `ResultsPerTargetF.i:12-99`.
//! This is where the reference implementation spends 79% of its per-target wall
//! time — only 21% is inside a PLS fit — so it is the part worth porting, and
//! the part most likely to diverge silently.
//!
//! Three ordering rules are load-bearing, because coefficient names are parsed
//! back out by `RegulationPerCondition`:
//!
//! 1. omic blocks are concatenated in **alphabetical** omic order, because R
//!    builds them with `split()`, which sorts by factor level — not in the
//!    order the omics were given on the command line;
//! 2. within a block, all plain regulator columns come first, then all
//!    interaction columns;
//! 3. interaction columns are **regulator-major**: `Group_a:R1, Group_b:R1,
//!    Group_a:R2, ...`, because R's `sapply` over regulators produces a matrix
//!    that `paste(collapse="+")` flattens column-major.

use crate::data::Association;
use crate::matrix::{sd, Mat};
use crate::prep::{Filter, Omic};
use std::collections::BTreeMap;

/// One regulator's row in a target's `allRegulators` table.
#[derive(Clone, Debug)]
pub struct RegulatorRow {
    pub regulator: String,
    pub omic: String,
    pub area: String,
    pub filter: Filter,
}

/// The assembled per-target design.
pub struct Design {
    /// Column labels, matching R's `colnames(des.mat2)[-1]` exactly.
    pub columns: Vec<String>,
    /// Scaled design, samples x columns.
    pub x: Mat,
    /// Every regulator associated with this target, with its disposition.
    pub regulators: Vec<RegulatorRow>,
}

/// `GetAllReg` for one target: the regulators associated with it, per omic.
/// With no association table for an omic, every regulator in that omic applies.
///
/// The reference implementation rescans the whole association table per target,
/// which is O(targets x associations); `Omic::by_target` makes this a lookup.
/// Same result, and it is one of the two O(n^2) scans that make the R version
/// superlinear in target count.
pub fn all_regulators(target: &str, omics: &[Omic]) -> Vec<RegulatorRow> {
    let mut out = Vec::new();
    for omic in omics {
        match &omic.associations {
            Some(_) => {
                if let Some(rows) = omic.by_target.get(target) {
                    // Areas for a repeated (target, regulator) pair are merged
                    // with ";" — R's `uniquePaste` after `aggregate`.
                    let mut merged: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
                    let mut order: Vec<&str> = Vec::new();
                    for a in rows {
                        if !merged.contains_key(a.regulator.as_str()) {
                            order.push(a.regulator.as_str());
                        }
                        let areas = merged.entry(a.regulator.as_str()).or_default();
                        if !a.area.is_empty() && !areas.contains(&a.area.as_str()) {
                            areas.push(a.area.as_str());
                        }
                    }
                    for reg in order {
                        out.push(RegulatorRow {
                            regulator: reg.to_string(),
                            omic: omic.name.clone(),
                            area: merged[reg].join(";"),
                            filter: Filter::Model,
                        });
                    }
                }
            }
            None => {
                for reg in &omic.data.row_names {
                    out.push(RegulatorRow {
                        regulator: reg.clone(),
                        omic: omic.name.clone(),
                        area: String::new(),
                        filter: Filter::Model,
                    });
                }
            }
        }
    }
    out
}

/// `RemovedRegulators`: label each regulator Model / MissingValue /
/// LowVariation, and collect the values of the Model ones.
pub fn classify(rows: &mut [RegulatorRow], omics: &[Omic]) {
    for row in rows.iter_mut() {
        let Some(omic) = omics.iter().find(|o| o.name == row.omic) else {
            continue;
        };
        if omic.removed_na.contains(&row.regulator) {
            row.filter = Filter::MissingValue;
        } else if omic.removed_lv.contains(&row.regulator) {
            row.filter = Filter::LowVariation;
        } else if !omic.data.row_names.contains(&row.regulator) {
            // Present in the association table but absent from the data matrix
            // after filtering: R drops it from the model silently.
            row.filter = Filter::MissingValue;
        }
    }
}

/// Assemble the scaled design matrix for one target.
///
/// `design_cols`/`design_values` are the `Group_*` indicator columns; pass an
/// empty slice for a run without a condition file. `interactions` mirrors
/// MORE's `interactions` argument, which `runMORE.R` leaves at its default
/// `TRUE` — with it off, `RegulationPerCondition` has no per-condition terms to
/// read back and the rpc table comes out empty.
pub fn build(
    rows: &[RegulatorRow],
    omics: &[Omic],
    design_cols: &[String],
    design_values: &[Vec<f64>],
    n_samples: usize,
    interactions: bool,
) -> Option<(Vec<String>, Mat)> {
    // Group the Model regulators by omic, alphabetically, as R's split() does.
    let mut blocks: BTreeMap<&str, Vec<&RegulatorRow>> = BTreeMap::new();
    for row in rows.iter().filter(|r| r.filter == Filter::Model) {
        blocks.entry(row.omic.as_str()).or_default().push(row);
    }
    if blocks.is_empty() {
        return None;
    }

    let mut columns: Vec<String> = Vec::new();
    let mut data: Vec<Vec<f64>> = Vec::new();

    for (omic_name, regs) in &blocks {
        let omic = omics.iter().find(|o| o.name == *omic_name)?;
        let index = omic.data.row_index();

        let mut block_cols: Vec<String> = Vec::new();
        let mut block_data: Vec<Vec<f64>> = Vec::new();

        for reg in regs {
            let Some(&r) = index.get(reg.regulator.as_str()) else {
                continue;
            };
            block_cols.push(reg.regulator.clone());
            block_data.push(omic.data.values[r].clone());
        }
        let plain = block_data.len();

        // Interaction terms: the product of a Group indicator and a regulator,
        // regulator-major so the names line up with R's formula expansion.
        if interactions && !design_cols.is_empty() {
            for j in 0..plain {
                for (c, cond) in design_cols.iter().enumerate() {
                    let values: Vec<f64> = (0..n_samples)
                        .map(|s| design_values[s][c] * block_data[j][s])
                        .collect();
                    block_cols.push(format!("{cond}:{}", block_cols[j]));
                    block_data.push(values);
                }
            }
        }

        // RegulatorsInteractions drops zero-variance columns inside the block
        // before the block is handed on.
        let keep: Vec<usize> = (0..block_cols.len())
            .filter(|&j| {
                let s = sd_na_rm(&block_data[j]);
                s.is_nan() || s > 0.0
            })
            .collect();

        for j in keep {
            // Each column is centred and unit-scaled individually; scaleType
            // "auto" adds no block reweighting, so Scaling.type is a concat.
            let mut col = block_data[j].clone();
            crate::matrix::scale_in_place(&mut col);
            columns.push(block_cols[j].clone());
            data.push(col);
        }
    }

    if columns.is_empty() {
        return None;
    }

    // The design columns are prepended, scaled the same way.
    let mut all_cols: Vec<String> = Vec::new();
    let mut all_data: Vec<Vec<f64>> = Vec::new();
    for (c, name) in design_cols.iter().enumerate() {
        let mut col: Vec<f64> = (0..n_samples).map(|s| design_values[s][c]).collect();
        crate::matrix::scale_in_place(&mut col);
        all_cols.push(name.clone());
        all_data.push(col);
    }
    all_cols.extend(columns);
    all_data.extend(data);

    // Final constant-column filter. R keeps a column whose sd is NA
    // (`is.na(sdNo0) | sdNo0 > 0`), so an all-NA column survives into the fit;
    // reproduced deliberately rather than tidied away.
    let keep: Vec<usize> = (0..all_cols.len())
        .filter(|&j| {
            let s = sd(&all_data[j]);
            s.is_nan() || s > 0.0
        })
        .collect();
    if keep.is_empty() {
        return None;
    }

    let cols: Vec<String> = keep.iter().map(|&j| all_cols[j].clone()).collect();
    let mat = Mat::from_columns(&keep.iter().map(|&j| all_data[j].clone()).collect::<Vec<_>>());
    Some((cols, mat))
}

/// `sd(x, na.rm = TRUE)` — used inside `RegulatorsInteractions`, unlike the
/// NA-propagating `sd` used by the final constant filter.
fn sd_na_rm(x: &[f64]) -> f64 {
    let v: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    sd(&v)
}

/// Recover regulator names from a significant *variable* name: an interaction
/// term `Group_Treat:TF-1` credits regulator `TF-1`. R does this with
/// `strsplit(v, ":", fixed = TRUE)` then intersects with the known regulators.
pub fn regulators_of(variable: &str, known: &[RegulatorRow]) -> Vec<String> {
    variable
        .split(':')
        .filter(|part| known.iter().any(|r| r.regulator == *part))
        .map(|s| s.to_string())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Frame;
    use std::collections::{HashMap, HashSet};

    fn omic(name: &str, regs: &[(&str, &[f64])], assoc: Option<Vec<Association>>) -> Omic {
        let data = Frame {
            row_names: regs.iter().map(|(n, _)| n.to_string()).collect(),
            col_names: (0..regs[0].1.len()).map(|i| format!("S{i}")).collect(),
            values: regs.iter().map(|(_, v)| v.to_vec()).collect(),
        };
        let mut by_target: HashMap<String, Vec<Association>> = HashMap::new();
        if let Some(a) = &assoc {
            for row in a {
                by_target.entry(row.target.clone()).or_default().push(row.clone());
            }
        }
        Omic {
            name: name.to_string(),
            input_data: data.clone(),
            data,
            associations: assoc,
            omic_type: 0,
            removed_na: HashSet::new(),
            removed_lv: HashSet::new(),
            by_target,
        }
    }

    fn assoc(t: &str, r: &str, area: &str) -> Association {
        Association { target: t.into(), regulator: r.into(), area: area.into() }
    }

    #[test]
    fn associations_select_only_the_targets_regulators() {
        let o = omic(
            "TF",
            &[("R1", &[1., 2., 3., 4.]), ("R2", &[4., 3., 2., 1.])],
            Some(vec![assoc("G1", "R1", "")]),
        );
        let rows = all_regulators("G1", &[o]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].regulator, "R1");
    }

    #[test]
    fn no_association_table_means_every_regulator_applies() {
        let o = omic("TF", &[("R1", &[1., 2., 3., 4.]), ("R2", &[4., 3., 2., 1.])], None);
        assert_eq!(all_regulators("G1", &[o]).len(), 2);
    }

    #[test]
    fn repeated_pairs_merge_their_areas_with_semicolons() {
        let o = omic(
            "TF",
            &[("R1", &[1., 2., 3., 4.])],
            Some(vec![assoc("G1", "R1", "PROMOTER"), assoc("G1", "R1", "1st_EXON")]),
        );
        let rows = all_regulators("G1", &[o]);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].area, "PROMOTER;1st_EXON");
    }

    #[test]
    fn filtered_regulators_are_labelled_not_dropped() {
        let mut o = omic("TF", &[("R1", &[1., 2., 3., 4.])], Some(vec![assoc("G1", "R2", "")]));
        o.removed_lv.insert("R2".to_string());
        let mut rows = all_regulators("G1", &[o]);
        let o2 = omic("TF", &[("R1", &[1., 2., 3., 4.])], None);
        let mut o2 = o2;
        o2.removed_lv.insert("R2".to_string());
        classify(&mut rows, &[o2]);
        assert_eq!(rows[0].filter, Filter::LowVariation);
    }

    #[test]
    fn interaction_columns_are_regulator_major() {
        let o = omic("TF", &[("R1", &[1., 2., 3., 4.]), ("R2", &[2., 1., 4., 3.])], None);
        let rows = all_regulators("G1", &[o]);
        let o = omic("TF", &[("R1", &[1., 2., 3., 4.]), ("R2", &[2., 1., 4., 3.])], None);
        let cols = vec!["Group_A".to_string(), "Group_B".to_string()];
        let dv = vec![vec![1., 0.], vec![1., 0.], vec![0., 1.], vec![0., 1.]];
        let (names, _) = build(&rows, &[o], &cols, &dv, 4, true).unwrap();
        // design columns, then R1, R2, then interactions regulator-major.
        assert_eq!(
            names,
            vec![
                "Group_A", "Group_B", "R1", "R2",
                "Group_A:R1", "Group_B:R1", "Group_A:R2", "Group_B:R2"
            ]
        );
    }

    #[test]
    fn interactions_can_be_switched_off() {
        let o = omic("TF", &[("R1", &[1., 2., 3., 4.])], None);
        let rows = all_regulators("G1", &[o]);
        let o = omic("TF", &[("R1", &[1., 2., 3., 4.])], None);
        let cols = vec!["Group_A".to_string(), "Group_B".to_string()];
        let dv = vec![vec![1., 0.], vec![1., 0.], vec![0., 1.], vec![0., 1.]];
        let (names, _) = build(&rows, &[o], &cols, &dv, 4, false).unwrap();
        assert_eq!(names, vec!["Group_A", "Group_B", "R1"]);
    }

    #[test]
    fn omic_blocks_are_concatenated_alphabetically_not_in_input_order() {
        let z = omic("zTF", &[("Z1", &[1., 2., 3., 4.])], None);
        let a = omic("aMi", &[("A1", &[4., 1., 3., 2.])], None);
        let rows = all_regulators("G1", &[z, a]);
        let z = omic("zTF", &[("Z1", &[1., 2., 3., 4.])], None);
        let a = omic("aMi", &[("A1", &[4., 1., 3., 2.])], None);
        let (names, _) = build(&rows, &[z, a], &[], &[], 4, false).unwrap();
        assert_eq!(names, vec!["A1", "Z1"]);
    }

    #[test]
    fn constant_columns_are_dropped() {
        let o = omic("TF", &[("R1", &[1., 2., 3., 4.]), ("FLAT", &[5., 5., 5., 5.])], None);
        let rows = all_regulators("G1", &[o]);
        let o = omic("TF", &[("R1", &[1., 2., 3., 4.]), ("FLAT", &[5., 5., 5., 5.])], None);
        let (names, _) = build(&rows, &[o], &[], &[], 4, false).unwrap();
        assert_eq!(names, vec!["R1"]);
    }

    #[test]
    fn scaled_columns_have_zero_mean_and_unit_sd() {
        let o = omic("TF", &[("R1", &[1., 2., 3., 4.])], None);
        let rows = all_regulators("G1", &[o]);
        let o = omic("TF", &[("R1", &[1., 2., 3., 4.])], None);
        let (_, x) = build(&rows, &[o], &[], &[], 4, false).unwrap();
        let col = x.col(0);
        assert!((col.iter().sum::<f64>()).abs() < 1e-12);
        assert!((sd(col) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_target_with_no_model_regulators_has_no_design() {
        let mut o = omic("TF", &[("R1", &[1., 2., 3., 4.])], None);
        o.removed_lv.insert("R1".to_string());
        let mut rows = all_regulators("G1", &[omic("TF", &[("R1", &[1., 2., 3., 4.])], None)]);
        classify(&mut rows, &[o]);
        let o = omic("TF", &[("R1", &[1., 2., 3., 4.])], None);
        assert!(build(&rows, &[o], &[], &[], 4, false).is_none());
    }

    #[test]
    fn an_interaction_term_credits_its_regulator() {
        let known = vec![RegulatorRow {
            regulator: "TF-1".into(),
            omic: "TF".into(),
            area: String::new(),
            filter: Filter::Model,
        }];
        assert_eq!(regulators_of("Group_Treat:TF-1", &known), vec!["TF-1"]);
        assert_eq!(regulators_of("TF-1", &known), vec!["TF-1"]);
        assert!(regulators_of("Group_Treat", &known).is_empty());
    }
}
