//! `CollinearityFilter1` — collapsing groups of correlated regulators.
//!
//! `GetMLR` passes `col.filter = "cor"`, so this is the filter on the
//! PaintOmics path (`ResultsPerTargetF.i.mlr:76`); `CollinearityFilter2`
//! (`"pcor"`, partial correlations) is unreachable from `runMORE.R` and is not
//! ported. Runs only when a target has more than one Model regulator.
//!
//! This is what the MLR recall gap was: the elastic net selects one *group
//! representative*, and `ResultsPerTargetF.i.mlr:170-179` then expands it back
//! to **every** regulator in that group. Without the grouping, a lasso that
//! picks 2 variables reports 2 edges where R reports a dozen. See `SPEC.md` §4.5.
//!
//! # Representative choice is random in R, deterministic here
//!
//! R picks the survivor with `sample(correlacionados, 1)`, a third RNG site on
//! the MLR path. It is the benign one: group *membership* is deterministic — a
//! clique in the correlation graph — and the expansion returns every member
//! whichever one is drawn, so the reported edge set does not depend on the draw.
//! The retained column's name and therefore the coefficient attribution do.
//! This port takes the first member in canonical order, which keeps edge sets
//! comparable while leaving per-condition coefficients for a collapsed group
//! attached to a possibly different member — a stated mechanism, not a
//! tolerance.

use crate::design::RegulatorRow;
use crate::matrix::sd;
use crate::prep::{Filter, Omic};

/// A collapsed clique of mutually correlated regulators.
#[derive(Clone, Debug)]
pub struct Group {
    /// `<omic>_mc<i>_R`, the name R gives the surviving column.
    pub name: String,
    pub representative: String,
    /// Every regulator in the clique, representative included.
    pub members: Vec<String>,
}

/// Pearson correlation. Under `scaleType = "auto"` R correlates the scaled
/// matrix, and Pearson is scale-invariant, so scaling is a no-op here.
fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let cov: f64 = a.iter().zip(b).map(|(x, y)| (x - ma) * (y - mb)).sum::<f64>() / (n - 1.0);
    let d = sd(a) * sd(b);
    if !(d > 0.0) {
        f64::NAN
    } else {
        cov / d
    }
}

/// Find the cliques of correlated Model regulators.
///
/// A component of the `|r| >= threshold` graph is collapsed **only if it is a
/// complete clique** (`nedges == csize*(csize-1)/2`). A merely connected
/// component is left alone — that check is in the R and dropping it would
/// over-collapse.
///
/// **Limitation, deliberate and reported.** R chooses the correlation *kind*
/// by omic-type pair: Pearson for numeric/numeric, `ltm::biserial.cor` for
/// numeric/binary, `psych::phi` for binary/binary. Only the numeric/numeric
/// case is implemented. Pairs involving an omic that `isBin` flagged binary are
/// **not grouped at all**, which is conservative — it can only under-collapse,
/// never invent a group — and is surfaced by the caller rather than hidden.
pub fn find_groups(
    rows: &[RegulatorRow],
    omics: &[Omic],
    threshold: f64,
) -> (Vec<Group>, usize) {
    let model: Vec<&RegulatorRow> = rows.iter().filter(|r| r.filter == Filter::Model).collect();
    if model.len() < 2 {
        return (Vec::new(), 0);
    }

    // Values per Model regulator, and whether its omic is binary.
    // A regulator that cannot be located is skipped, NOT treated as grounds to
    // abandon grouping altogether. Aborting on the first miss silently disabled
    // the entire filter and was the reason this found zero cliques on data
    // where R finds six.
    let mut model_ok: Vec<&RegulatorRow> = Vec::with_capacity(model.len());
    let mut values: Vec<&[f64]> = Vec::with_capacity(model.len());
    let mut binary: Vec<bool> = Vec::with_capacity(model.len());
    for r in &model {
        let Some(omic) = omics.iter().find(|o| o.name == r.omic) else {
            continue;
        };
        let Some(&idx) = omic.data.row_index().get(r.regulator.as_str()) else {
            continue;
        };
        model_ok.push(r);
        values.push(&omic.data.values[idx]);
        binary.push(omic.omic_type == 1);
    }
    let model = model_ok;
    if model.len() < 2 {
        return (Vec::new(), 0);
    }

    let n = model.len();
    let mut adj = vec![vec![false; n]; n];
    let mut corr = vec![vec![0.0f64; n]; n];
    let mut skipped_binary = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if binary[i] || binary[j] {
                skipped_binary += 1;
                continue;
            }
            let r = pearson(values[i], values[j]);
            if r.is_nan() {
                continue;
            }
            corr[i][j] = r;
            corr[j][i] = r;
            if r.abs() >= threshold {
                adj[i][j] = true;
                adj[j][i] = true;
            }
        }
    }

    // Connected components.
    let mut component = vec![usize::MAX; n];
    let mut n_components = 0;
    for start in 0..n {
        if component[start] != usize::MAX {
            continue;
        }
        let mut stack = vec![start];
        component[start] = n_components;
        while let Some(v) = stack.pop() {
            for w in 0..n {
                if adj[v][w] && component[w] == usize::MAX {
                    component[w] = n_components;
                    stack.push(w);
                }
            }
        }
        n_components += 1;
    }

    let mut groups = Vec::new();
    let mut collapsed = 0usize;
    for c in 0..n_components {
        let members: Vec<usize> = (0..n).filter(|&i| component[i] == c).collect();
        if members.len() < 2 {
            continue;
        }
        // Complete-clique test, exactly as in the R.
        let mut edges = 0usize;
        for a in 0..members.len() {
            for b in (a + 1)..members.len() {
                if adj[members[a]][members[b]] {
                    edges += 1;
                }
            }
        }
        if edges == members.len() * (members.len() - 1) / 2 {
            // Complete clique: collapse whole.
            collapsed += 1;
            let rep = members[0];
            groups.push(Group {
                name: format!("{}_mc{}_R", model[rep].omic, collapsed),
                representative: model[rep].regulator.clone(),
                members: members.iter().map(|&i| model[i].regulator.clone()).collect(),
            });
            continue;
        }

        // Not a clique. R does NOT discard it -- it peels stars until every
        // node is isolated: take the highest-degree node, absorb all of its
        // neighbours into one group, remove them, repeat. This is the branch
        // that turns a single 20-node component with 21 edges into six groups,
        // and its naming `<omic>_mc<i>_<j>_R` is what the R output shows.
        let mut alive: Vec<bool> = vec![false; n];
        for &m in &members {
            alive[m] = true;
        }
        let mut j = 0usize;
        loop {
            let degree = |v: usize, alive: &[bool]| -> usize {
                (0..n).filter(|&w| alive[w] && adj[v][w]).count()
            };
            let mut best: Option<usize> = None;
            let mut best_deg = 0usize;
            let mut best_sum = f64::NEG_INFINITY;
            // How many nodes tie on BOTH degree and summed |r| — the point at
            // which R falls back to sample() and this port cannot follow.
            let mut tied = 0usize;
            for &v in &members {
                if !alive[v] {
                    continue;
                }
                let d = degree(v, &alive);
                if d == 0 {
                    continue;
                }
                // Tie-break on the summed absolute correlation of the node's
                // edges, as R does; R breaks a further tie with sample(), this
                // takes the first in canonical order.
                let sum: f64 =
                    (0..n).filter(|&w| alive[w] && adj[v][w]).map(|w| corr[v][w].abs()).sum();
                if d > best_deg || (d == best_deg && sum > best_sum) {
                    best = Some(v);
                    best_deg = d;
                    best_sum = sum;
                    tied = 1;
                } else if d == best_deg && (sum - best_sum).abs() < 1e-12 {
                    tied += 1;
                }
            }
            let Some(rep) = best else { break };
            if tied > 1 && std::env::var_os("MORE_RS_DEBUG_MLR").is_some() {
                eprintln!("DBG   TIE degree={best_deg} among {tied} nodes -- R would sample()");
            }

            let neighbours: Vec<usize> =
                (0..n).filter(|&w| alive[w] && adj[rep][w]).collect();
            j += 1;
            let mut group_members = vec![model[rep].regulator.clone()];
            for &w in &neighbours {
                group_members.push(model[w].regulator.clone());
                alive[w] = false;
            }
            alive[rep] = false;
            groups.push(Group {
                name: format!("{}_mc{}_{}_R", model[rep].omic, c + 1, j),
                representative: model[rep].regulator.clone(),
                members: group_members,
            });
        }
    }

    (groups, skipped_binary)
}

/// Regulators removed from the design because a group collapsed onto someone
/// else — everything in a group except its representative.
pub fn suppressed(groups: &[Group]) -> Vec<String> {
    groups
        .iter()
        .flat_map(|g| g.members.iter().filter(|m| **m != g.representative).cloned())
        .collect()
}

/// Expand a selected regulator set through the groups: a representative stands
/// for every member of its clique (`ResultsPerTargetF.i.mlr:170-179`).
pub fn expand(selected: &[String], groups: &[Group]) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for reg in selected {
        match groups.iter().find(|g| g.representative == *reg) {
            Some(g) => {
                for m in &g.members {
                    if !out.contains(m) {
                        out.push(m.clone());
                    }
                }
            }
            None => {
                if !out.contains(reg) {
                    out.push(reg.clone());
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::Frame;
    use std::collections::{HashMap, HashSet};

    fn omic(name: &str, regs: &[(&str, Vec<f64>)], binary: bool) -> Omic {
        Omic {
            name: name.into(),
            data: Frame {
                row_names: regs.iter().map(|(n, _)| n.to_string()).collect(),
                col_names: (0..regs[0].1.len()).map(|i| format!("S{i}")).collect(),
                values: regs.iter().map(|(_, v)| v.clone()).collect(),
            },
            associations: None,
            omic_type: if binary { 1 } else { 0 },
            removed_na: HashSet::new(),
            removed_lv: HashSet::new(),
            by_target: HashMap::new(),
        }
    }

    fn row(name: &str, omic: &str) -> RegulatorRow {
        RegulatorRow {
            regulator: name.into(),
            omic: omic.into(),
            area: String::new(),
            filter: Filter::Model,
        }
    }

    #[test]
    fn perfectly_correlated_regulators_form_one_group() {
        let base = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let o = omic(
            "TF",
            &[
                ("A", base.clone()),
                ("B", base.iter().map(|v| 2.0 * v).collect()),
                ("C", base.iter().map(|v| 3.0 * v + 1.0).collect()),
            ],
            false,
        );
        let rows = vec![row("A", "TF"), row("B", "TF"), row("C", "TF")];
        let (groups, _) = find_groups(&rows, &[o], 0.7);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].members.len(), 3);
        assert_eq!(groups[0].name, "TF_mc1_R");
    }

    #[test]
    fn uncorrelated_regulators_are_not_grouped() {
        let o = omic(
            "TF",
            &[
                ("A", vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
                ("B", vec![1.0, 1.0, -1.0, -1.0, 1.0, 1.0]),
            ],
            false,
        );
        let rows = vec![row("A", "TF"), row("B", "TF")];
        let (groups, _) = find_groups(&rows, &[o], 0.7);
        assert!(groups.is_empty());
    }

    #[test]
    fn negative_correlation_still_groups() {
        let base = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let o = omic(
            "TF",
            &[("A", base.clone()), ("B", base.iter().map(|v| -v).collect())],
            false,
        );
        let rows = vec![row("A", "TF"), row("B", "TF")];
        let (groups, _) = find_groups(&rows, &[o], 0.7);
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn a_connected_but_incomplete_component_is_left_alone() {
        // A~B and B~C correlate, A~C does not: connected, not a clique, so R
        // does not collapse it and neither may this.
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let c = vec![8.0, 6.0, 9.0, 1.0, 3.0, 2.0, 7.0, 4.0];
        let b: Vec<f64> = a.iter().zip(&c).map(|(x, y)| x + y).collect();
        let o = omic("TF", &[("A", a), ("B", b), ("C", c)], false);
        let rows = vec![row("A", "TF"), row("B", "TF"), row("C", "TF")];
        let (groups, _) = find_groups(&rows, &[o], 0.7);
        for g in &groups {
            assert_eq!(g.members.len(), 2, "an incomplete component was collapsed");
        }
    }

    #[test]
    fn binary_omics_are_skipped_and_counted() {
        let o = omic(
            "TF",
            &[("A", vec![0.0, 1.0, 0.0, 1.0]), ("B", vec![0.0, 1.0, 0.0, 1.0])],
            true,
        );
        let rows = vec![row("A", "TF"), row("B", "TF")];
        let (groups, skipped) = find_groups(&rows, &[o], 0.7);
        assert!(groups.is_empty());
        assert_eq!(skipped, 1);
    }

    #[test]
    fn suppressed_lists_every_member_but_the_representative() {
        let g = Group {
            name: "TF_mc1_R".into(),
            representative: "A".into(),
            members: vec!["A".into(), "B".into(), "C".into()],
        };
        let mut s = suppressed(&[g]);
        s.sort();
        assert_eq!(s, vec!["B", "C"]);
    }

    #[test]
    fn selecting_a_representative_expands_to_the_whole_group() {
        let g = Group {
            name: "TF_mc1_R".into(),
            representative: "A".into(),
            members: vec!["A".into(), "B".into(), "C".into()],
        };
        let out = expand(&["A".to_string()], &[g]);
        assert_eq!(out, vec!["A", "B", "C"]);
    }

    #[test]
    fn an_ungrouped_selection_passes_through_unchanged() {
        let g = Group {
            name: "TF_mc1_R".into(),
            representative: "A".into(),
            members: vec!["A".into(), "B".into()],
        };
        assert_eq!(expand(&["Z".to_string()], &[g]), vec!["Z"]);
    }

    #[test]
    fn a_single_model_regulator_forms_no_group() {
        let o = omic("TF", &[("A", vec![1.0, 2.0, 3.0, 4.0])], false);
        let (groups, _) = find_groups(&[row("A", "TF")], &[o], 0.7);
        assert!(groups.is_empty());
    }
}
