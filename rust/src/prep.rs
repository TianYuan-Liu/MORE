//! Whole-run preparation: name mangling, global filters, grouping.
//!
//! Everything here runs **once per `more()` call, over all targets**. That is
//! why splitting a run into chunks is not semantically free: the automatic
//! low-variation threshold is a fraction of the maximum variability observed
//! across the whole call, so a chunked run keeps a different set of regulators.
//! Measured on the R side, chunking 1000 targets into 8 pieces is 4.6x faster
//! precisely because it shrinks this scope — any such mitigation has to be
//! validated by the equivalence harness, never assumed.
//!
//! Ports `GetPLS:118-277` and `LowVariationRegu`/`LowVariatFilter`.

use crate::cli::MinVariation;
use crate::data::{Association, Frame};
use std::collections::{BTreeMap, HashMap, HashSet};

/// Fraction of missing values above which a regulator or sample is dropped.
/// MORE's `percNA` default, which `runMORE.R` never overrides.
pub const PERC_NA: f64 = 0.2;

/// Why a regulator is not in a target's model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Filter {
    Model,
    MissingValue,
    LowVariation,
    Constant,
}

impl Filter {
    pub fn as_str(self) -> &'static str {
        match self {
            Filter::Model => "Model",
            Filter::MissingValue => "MissingValue",
            Filter::LowVariation => "LowVariation",
            Filter::Constant => "Constant",
        }
    }
}

/// One regulatory omic after preparation.
pub struct Omic {
    pub name: String,
    /// The modelling matrix: high-NA and low-variation regulators removed.
    pub data: Frame,
    /// The matrix as read from disk, column-subset to the common samples and
    /// nothing else — R's `regulatoryData[[name]]`. Its regulator filters never
    /// touch that object because they run inside MORE, on MORE's own copy.
    ///
    /// The values and association files are built from this, not from `data`,
    /// so a regulator MORE refused to model still contributes its
    /// `GENE:::REGULATOR` rows to PA Step 1. Significance only drives the
    /// yellow-star overlay.
    pub input_data: Frame,
    pub associations: Option<Vec<Association>>,
    /// 0 = numeric, 1 = binary — MORE's `omicType`, inferred by `isBin`.
    pub omic_type: u8,
    pub removed_na: HashSet<String>,
    pub removed_lv: HashSet<String>,
    /// target -> regulators, built once so the per-target loop is a lookup
    /// rather than R's linear scan of the whole association table (`GetAllReg`
    /// is O(n^2) over targets in the reference implementation).
    pub by_target: HashMap<String, Vec<Association>>,
}

/// `isBin`: an omic is binary when both the first column and the first row
/// contain exactly two distinct non-missing values.
pub fn is_binary(f: &Frame) -> u8 {
    if f.nrow() == 0 || f.ncol() == 0 {
        return 0;
    }
    let col: HashSet<u64> = f
        .values
        .iter()
        .map(|r| r[0])
        .filter(|v| !v.is_nan())
        .map(|v| v.to_bits())
        .collect();
    let row: HashSet<u64> = f.values[0]
        .iter()
        .filter(|v| !v.is_nan())
        .map(|v| v.to_bits())
        .collect();
    if col.len() == 2 && row.len() == 2 {
        1
    } else {
        0
    }
}

/// Rewrite regulator IDs the way `GetPLS:120-134` does, before anything else
/// looks at them. `:` is the interaction separator, and the `_R`/`_P`/`_N`
/// suffixes collide with the collinearity-group markers.
pub fn mangle_id(id: &str) -> String {
    let mut s = id.replace(':', "-");
    for suffix in ["_R", "_P", "_N"] {
        if s.ends_with(suffix) {
            s.truncate(s.len() - 2);
            s.push('-');
            s.push(suffix.chars().nth(1).unwrap());
            break;
        }
    }
    s
}

/// Condition label per sample: the condition row pasted with `_`.
pub fn group_labels(condition: &Frame) -> Vec<String> {
    condition
        .values
        .iter()
        .map(|row| {
            row.iter()
                .map(|v| format_r_number(*v))
                .collect::<Vec<_>>()
                .join("_")
        })
        .collect()
}

/// R prints a whole-valued double without a decimal point, which is what ends
/// up inside the `Group_1_0` style column names.
pub fn format_r_number(v: f64) -> String {
    if v.is_nan() {
        return "NA".into();
    }
    if v == v.trunc() && v.abs() < 1e15 {
        format!("{}", v as i64)
    } else {
        format!("{v}")
    }
}

/// `model.matrix` column names, in R's factor-level order (alphabetical),
/// prefixed `Group_`.
///
/// The two methods build this differently and the difference is not cosmetic:
///
/// * PLS1 (`MORE_PLS.R:337`) uses `model.matrix(~0 + ., ...)` — no intercept,
///   so **every** level gets a column;
/// * MLR (`MORE_MLR.R:318`) uses `model.matrix(~Group)[, -1, drop = FALSE]` —
///   an intercept model with the first (alphabetically first) level dropped as
///   the reference.
///
/// Giving MLR all the levels hands the elastic net a design that is rank
/// deficient by construction, doubles the interaction terms, and splits every
/// condition effect across two collinear columns. On a two-condition run that
/// is 26 design columns where R has 17.
pub fn design_columns(groups: &[String], drop_reference: bool) -> Vec<String> {
    let levels: std::collections::BTreeSet<&String> = groups.iter().collect();
    let mut cols: Vec<String> = levels.into_iter().map(|g| format!("Group_{g}")).collect();
    if drop_reference && !cols.is_empty() {
        cols.remove(0);
    }
    cols
}

/// Condition column names **for the rpc table**, which are NOT the design
/// matrix's columns.
///
/// `RegulationPerCondition:346` builds them with `paste("Group", unique(Group))`,
/// and `unique()` preserves first-appearance order. `model.matrix` — which
/// produces the design columns and therefore the interaction term names — sorts
/// by factor level instead. The two orders coincide only by luck, so a run with
/// controls before treatments emits `Group_1_0, Group_0_1` in the rpc header
/// while the interaction terms are named in the sorted order. Both are needed.
pub fn rpc_columns(groups: &[String]) -> Vec<String> {
    let mut seen: Vec<&str> = Vec::new();
    for g in groups {
        if !seen.contains(&g.as_str()) {
            seen.push(g.as_str());
        }
    }
    seen.into_iter().map(|g| format!("Group_{g}")).collect()
}

/// The 0/1 design matrix itself: `design[sample][column]`.
pub fn design_matrix(groups: &[String], columns: &[String]) -> Vec<Vec<f64>> {
    groups
        .iter()
        .map(|g| {
            let mine = format!("Group_{g}");
            columns.iter().map(|c| if *c == mine { 1.0 } else { 0.0 }).collect()
        })
        .collect()
}

/// Per-regulator mean within each condition, in condition-label order —
/// R's `t(apply(x, 1, tapply, ExpGroups, mean))`.
fn condition_means(f: &Frame, groups: &[String]) -> Vec<Vec<f64>> {
    // BTreeMap so the condition order matches R's sorted factor levels.
    let mut order: BTreeMap<&str, Vec<usize>> = BTreeMap::new();
    for (i, g) in groups.iter().enumerate() {
        order.entry(g.as_str()).or_default().push(i);
    }
    f.values
        .iter()
        .map(|row| {
            order
                .values()
                .map(|idx| {
                    let vals: Vec<f64> = idx.iter().map(|&i| row[i]).collect();
                    vals.iter().sum::<f64>() / vals.len() as f64
                })
                .collect()
        })
        .collect()
}

/// `LowVariatFilter`. Returns the regulator IDs to drop.
///
/// Numeric omics under the automatic threshold use the standard deviation
/// across condition means and keep anything above 10% of the largest observed
/// sd. Under a user threshold they use the range and compare against the
/// threshold directly. Binary omics always use the range, against a tenth of
/// the maximum (automatic) or the threshold (user).
pub fn low_variation(
    f: &Frame,
    groups: &[String],
    omic_type: u8,
    min_variation: MinVariation,
) -> HashSet<String> {
    if f.nrow() == 0 {
        return HashSet::new();
    }
    let means = condition_means(f, groups);

    let stat: Vec<f64> = match (omic_type, min_variation) {
        (0, MinVariation::Auto) => means.iter().map(|m| crate::matrix::sd(m)).collect(),
        _ => means
            .iter()
            .map(|m| {
                let mut lo = f64::INFINITY;
                let mut hi = f64::NEG_INFINITY;
                for v in m {
                    if !v.is_nan() {
                        lo = lo.min(*v);
                        hi = hi.max(*v);
                    }
                }
                if lo.is_infinite() {
                    f64::NAN
                } else {
                    hi - lo
                }
            })
            .collect(),
    };

    let cutoff = match min_variation {
        // Numeric: 10% of the largest sd. Binary: a tenth of the largest range.
        MinVariation::Auto => {
            let max = stat.iter().filter(|v| !v.is_nan()).fold(f64::NEG_INFINITY, |a, b| a.max(*b));
            if max.is_infinite() {
                return HashSet::new();
            }
            if omic_type == 0 {
                max * (10.0 / 100.0)
            } else {
                max / 10.0
            }
        }
        MinVariation::Value(v) => v,
    };

    f.row_names
        .iter()
        .zip(&stat)
        .filter(|(_, s)| !(**s > cutoff))
        .map(|(n, _)| n.clone())
        .collect()
}

/// Regulators whose missing fraction exceeds `PERC_NA`.
pub fn high_na_rows(f: &Frame) -> HashSet<String> {
    f.row_names
        .iter()
        .zip(&f.values)
        .filter(|(_, row)| {
            let n = row.len().max(1) as f64;
            row.iter().filter(|v| v.is_nan()).count() as f64 / n > PERC_NA
        })
        .map(|(n, _)| n.clone())
        .collect()
}

/// Samples whose missing fraction exceeds `PERC_NA` in this omic. A sample
/// dropped by any omic is dropped from every matrix.
pub fn high_na_columns(f: &Frame) -> HashSet<String> {
    (0..f.ncol())
        .filter(|&c| {
            let n = f.nrow().max(1) as f64;
            f.values.iter().filter(|r| r[c].is_nan()).count() as f64 / n > PERC_NA
        })
        .map(|c| f.col_names[c].clone())
        .collect()
}

/// Why a target has no model at all.
#[derive(Clone, Debug)]
pub struct TargetProblem {
    pub target: String,
    pub problem: &'static str,
}

/// Apply the whole-run target filters in `GetPLS`'s order, returning the
/// surviving target IDs alongside the reasons for every exclusion.
pub fn filter_targets(
    target: &Frame,
    has_association_for: &dyn Fn(&str) -> bool,
    any_associations: bool,
) -> (Vec<String>, Vec<TargetProblem>) {
    let mut problems = Vec::new();
    let mut keep = Vec::new();

    for (i, name) in target.row_names.iter().enumerate() {
        let row = &target.values[i];

        if row.iter().any(|v| v.is_infinite()) {
            problems.push(TargetProblem { target: name.clone(), problem: "-Inf/Inf values" });
            continue;
        }
        // min.obs = ncol(targetData): a single missing value drops the target.
        if row.iter().any(|v| v.is_nan()) {
            problems
                .push(TargetProblem { target: name.clone(), problem: "Too many missing values" });
            continue;
        }
        if any_associations && !has_association_for(name) {
            problems.push(TargetProblem {
                target: name.clone(),
                problem: "Target feature had no initial regulators",
            });
            continue;
        }
        if !(crate::matrix::sd(row) > 0.0) {
            problems.push(TargetProblem {
                target: name.clone(),
                problem: "Response values are constant",
            });
            continue;
        }
        keep.push(name.clone());
    }

    (keep, problems)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(rows: &[(&str, &[f64])], cols: &[&str]) -> Frame {
        Frame {
            row_names: rows.iter().map(|(n, _)| n.to_string()).collect(),
            col_names: cols.iter().map(|c| c.to_string()).collect(),
            values: rows.iter().map(|(_, v)| v.to_vec()).collect(),
        }
    }

    #[test]
    fn colons_in_ids_become_hyphens() {
        assert_eq!(mangle_id("chr1:100"), "chr1-100");
    }

    #[test]
    fn trailing_marker_suffixes_are_rewritten() {
        assert_eq!(mangle_id("TF_R"), "TF-R");
        assert_eq!(mangle_id("TF_P"), "TF-P");
        assert_eq!(mangle_id("TF_N"), "TF-N");
    }

    #[test]
    fn a_suffix_in_the_middle_is_left_alone() {
        assert_eq!(mangle_id("TF_راw"), "TF_راw");
        assert_eq!(mangle_id("A_Rb"), "A_Rb");
    }

    #[test]
    fn group_labels_paste_the_condition_row() {
        let c = frame(&[("S1", &[1.0, 0.0]), ("S2", &[0.0, 1.0])], &["Ctrl", "Treat"]);
        assert_eq!(group_labels(&c), vec!["1_0", "0_1"]);
    }

    #[test]
    fn design_columns_are_sorted_like_r_factor_levels() {
        let g = vec!["1_0".to_string(), "0_1".to_string(), "1_0".to_string()];
        assert_eq!(design_columns(&g, false), vec!["Group_0_1", "Group_1_0"]);
    }

    #[test]
    fn rpc_columns_follow_first_appearance_not_sorted_order() {
        // Controls first, treatments second: R's unique() yields 1_0 then 0_1,
        // the opposite of the sorted factor levels the design matrix uses.
        let g = vec!["1_0".to_string(), "1_0".to_string(), "0_1".to_string()];
        assert_eq!(rpc_columns(&g), vec!["Group_1_0", "Group_0_1"]);
        assert_eq!(design_columns(&g, false), vec!["Group_0_1", "Group_1_0"]);
    }

    #[test]
    fn the_design_matrix_is_one_hot_per_sample() {
        let g = vec!["1_0".to_string(), "0_1".to_string()];
        let cols = design_columns(&g, false);
        let d = design_matrix(&g, &cols);
        assert_eq!(d[0], vec![0.0, 1.0]);
        assert_eq!(d[1], vec![1.0, 0.0]);
    }

    #[test]
    fn binary_omics_are_detected() {
        let f = frame(&[("R1", &[0.0, 1.0]), ("R2", &[1.0, 0.0])], &["S1", "S2"]);
        assert_eq!(is_binary(&f), 1);
    }

    #[test]
    fn numeric_omics_are_not_flagged_binary() {
        let f = frame(
            &[("R1", &[0.3, 1.7, 2.2]), ("R2", &[2.5, 0.1, 0.9]), ("R3", &[1.1, 0.4, 3.0])],
            &["S1", "S2", "S3"],
        );
        assert_eq!(is_binary(&f), 0);
    }

    #[test]
    fn is_binary_calls_any_two_sample_omic_binary() {
        // Not a port defect — `isBin` inspects only the first column and the
        // first row, so with two samples and two regulators any distinct values
        // give exactly two levels in both. Pinned because a job with two
        // samples silently takes the binary low-variation branch.
        let f = frame(&[("R1", &[0.3, 1.7]), ("R2", &[2.5, 0.1])], &["S1", "S2"]);
        assert_eq!(is_binary(&f), 1);
    }

    #[test]
    fn regulators_below_a_tenth_of_the_largest_sd_are_dropped() {
        // Condition means: R1 spans 0..10 (sd large), R2 is flat.
        let g = vec!["A".to_string(), "A".to_string(), "B".to_string(), "B".to_string()];
        let f = frame(
            &[("R1", &[0.0, 0.0, 10.0, 10.0]), ("R2", &[1.0, 1.0, 1.0, 1.0])],
            &["S1", "S2", "S3", "S4"],
        );
        let dropped = low_variation(&f, &g, 0, MinVariation::Auto);
        assert!(dropped.contains("R2"));
        assert!(!dropped.contains("R1"));
    }

    #[test]
    fn a_user_threshold_compares_the_range_directly() {
        let g = vec!["A".to_string(), "A".to_string(), "B".to_string(), "B".to_string()];
        let f = frame(
            &[("R1", &[0.0, 0.0, 10.0, 10.0]), ("R2", &[1.0, 1.0, 1.5, 1.5])],
            &["S1", "S2", "S3", "S4"],
        );
        // Range of R2's condition means is 0.5, below the threshold of 1.0.
        let dropped = low_variation(&f, &g, 0, MinVariation::Value(1.0));
        assert!(dropped.contains("R2"));
        assert!(!dropped.contains("R1"));
    }

    #[test]
    fn regulators_over_the_na_fraction_are_dropped() {
        // The test is strict `>`, so exactly 20% survives and 40% does not.
        let n = f64::NAN;
        let f = frame(
            &[("R1", &[n, n, 1.0, 1.0, 1.0]), ("R2", &[n, 1.0, 1.0, 1.0, 1.0])],
            &["a", "b", "c", "d", "e"],
        );
        let dropped = high_na_rows(&f);
        assert!(dropped.contains("R1"), "40% NA exceeds the 20% limit");
        assert!(!dropped.contains("R2"), "exactly 20% NA is not over the limit");
    }

    #[test]
    fn samples_over_the_na_fraction_are_dropped() {
        let n = f64::NAN;
        let f = frame(&[("R1", &[n, 1.0]), ("R2", &[n, 1.0]), ("R3", &[n, 1.0])], &["bad", "ok"]);
        let dropped = high_na_columns(&f);
        assert!(dropped.contains("bad"));
        assert!(!dropped.contains("ok"));
    }

    #[test]
    fn a_constant_target_is_excluded_with_a_reason() {
        let t = frame(&[("G1", &[1.0, 1.0, 1.0]), ("G2", &[1.0, 2.0, 3.0])], &["a", "b", "c"]);
        let (keep, problems) = filter_targets(&t, &|_| true, false);
        assert_eq!(keep, vec!["G2"]);
        assert_eq!(problems[0].problem, "Response values are constant");
    }

    #[test]
    fn a_target_with_any_missing_value_is_excluded() {
        let t = frame(&[("G1", &[1.0, f64::NAN, 3.0]), ("G2", &[1.0, 2.0, 3.0])], &["a", "b", "c"]);
        let (keep, problems) = filter_targets(&t, &|_| true, false);
        assert_eq!(keep, vec!["G2"]);
        assert_eq!(problems[0].problem, "Too many missing values");
    }

    #[test]
    fn an_infinite_target_is_excluded_before_the_na_test() {
        let t = frame(&[("G1", &[1.0, f64::INFINITY, 3.0])], &["a", "b", "c"]);
        let (keep, problems) = filter_targets(&t, &|_| true, false);
        assert!(keep.is_empty());
        assert_eq!(problems[0].problem, "-Inf/Inf values");
    }

    #[test]
    fn a_target_with_no_associations_is_excluded_when_associations_exist() {
        let t = frame(&[("G1", &[1.0, 2.0, 3.0]), ("G2", &[3.0, 1.0, 2.0])], &["a", "b", "c"]);
        let (keep, problems) = filter_targets(&t, &|n| n == "G1", true);
        assert_eq!(keep, vec!["G1"]);
        assert_eq!(problems[0].problem, "Target feature had no initial regulators");
    }
}
