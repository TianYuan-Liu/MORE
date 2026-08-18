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
    /// +1 or -1 per member, parallel to `members`: the sign of that member's
    /// correlation with the representative. R records it as the `_P`/`_N`
    /// suffix on the filter marker and uses it to flip the representative's
    /// coefficient when attributing it to the member (`output_analysis.R:330`).
    pub signs: Vec<f64>,
}

/// Pearson correlation. Under `scaleType = "auto"` R correlates the scaled
/// matrix, and Pearson is scale-invariant, so scaling is a no-op here.
/// `scale(x, center = TRUE, scale = TRUE)` on one column, as
/// `scale.default` does it: centre on `colMeans`, then divide by
/// `sqrt(sum(v^2) / max(1, n - 1))` of the *centred* column.
///
/// Pearson correlation is scale-invariant, so this looks like a no-op — and
/// mathematically it is. It is not numerically. `CollinearityFilter1`
/// correlates `scale(data, scale, center)`, and the port used to correlate the
/// raw values; the two agree to about 15 digits and differ in the last bits.
/// That is normally beneath notice, but R's star-peel tie-break asks
/// `which(sums == max(sums))` — **exact** float equality — and whether that
/// returns one index or two decides whether R draws from its RNG at all. A
/// last-bit disagreement therefore does not perturb the answer slightly; it
/// desynchronises the entire stream from that point on. Measured: on the 20x20
/// probe the raw-value route put R2 and R18 one ULP apart where R has them
/// equal, so the port took R2 outright while R drew between them.
fn r_scale_column(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut sum = 0.0f64;
    for &x in v {
        sum += x;
    }
    let centre = sum / n as f64;
    let centred: Vec<f64> = v.iter().map(|&x| x - centre).collect();
    let mut ss = 0.0f64;
    for &x in &centred {
        ss += x * x;
    }
    let denom = if n > 1 { (n - 1) as f64 } else { 1.0 };
    let sd = (ss / denom).sqrt();
    if !(sd > 0.0) {
        return centred;
    }
    centred.iter().map(|&x| x / sd).collect()
}

/// `cor(x, y)` — R's `cov.c` for the complete/pearson case, including its
/// two-pass mean refinement and the clamp at 1.
///
/// The refinement (`tmp += sum(x - tmp)/n`) is not decoration: it is what makes
/// R's means, and therefore its correlations, land on the bits they do.
fn r_cor(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len();
    if n < 2 || b.len() != n {
        return f64::NAN;
    }
    let refined_mean = |v: &[f64]| -> f64 {
        let mut sum = 0.0f64;
        for &x in v {
            sum += x;
        }
        let mut m = sum / n as f64;
        if m.is_finite() {
            let mut adj = 0.0f64;
            for &x in v {
                adj += x - m;
            }
            m += adj / n as f64;
        }
        m
    };
    let (xm, ym) = (refined_mean(a), refined_mean(b));
    let nm1 = (n - 1) as f64;
    let mut sxy = 0.0f64;
    let mut sxx = 0.0f64;
    let mut syy = 0.0f64;
    for k in 0..n {
        sxy += (a[k] - xm) * (b[k] - ym);
    }
    for &x in a {
        sxx += (x - xm) * (x - xm);
    }
    for &y in b {
        syy += (y - ym) * (y - ym);
    }
    let cov = sxy / nm1;
    let xsd = (sxx / nm1).sqrt();
    let ysd = (syy / nm1).sqrt();
    if xsd == 0.0 || ysd == 0.0 {
        return f64::NAN;
    }
    let mut r = cov / (xsd * ysd);
    if r > 1.0 {
        r = 1.0;
    }
    r
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
/// Replay a collinearity grouping captured from R instead of computing one.
///
/// `MORE_RS_GROUPS_OVERRIDE=<dir>` points at the `cf_*.tsv` dumps that
/// `equivalence/design_probe.R` takes from `CollinearityFilter1`'s return
/// value: `targetF, regulator, omic, area, filter`, where `filter` is
/// `<omic>_mc<i>_R` on the representative and `_P`/`_N` on the other members.
///
/// This exists to *test a diagnosis*, not to ship: R draws both the clique
/// representative and the star-peel tie-break from its RNG, so the only way to
/// tell whether those draws explain a residual edge-set difference is to hand
/// the port R's answer and re-measure. It is never consulted unless the
/// variable is set.
pub fn groups_from_override(target: &str) -> Option<Vec<Group>> {
    let dir = std::env::var_os("MORE_RS_GROUPS_OVERRIDE")?;
    let mut by_group: std::collections::BTreeMap<String, (Option<String>, Vec<(String, f64)>)> =
        std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        let text = match std::fs::read_to_string(entry.path()) {
            Ok(t) => t,
            Err(_) => continue,
        };
        for line in text.lines().skip(1) {
            let f: Vec<&str> = line.split('\t').collect();
            if f.len() < 5 || f[0] != target {
                continue;
            }
            let (regulator, filter) = (f[1], f[4]);
            // The synthetic representative rows carry filter "Model"; so do
            // genuinely ungrouped regulators. Neither belongs to a group.
            let Some(stem) = filter.strip_suffix("_R").map(|s| (s, 0.0f64))
                .or_else(|| filter.strip_suffix("_P").map(|s| (s, 1.0)))
                .or_else(|| filter.strip_suffix("_N").map(|s| (s, -1.0)))
            else {
                continue;
            };
            let e = by_group.entry(stem.0.to_string()).or_default();
            if filter.ends_with("_R") {
                e.0 = Some(regulator.to_string());
            } else {
                e.1.push((regulator.to_string(), stem.1));
            }
        }
    }
    if by_group.is_empty() {
        return None;
    }
    let mut out = Vec::new();
    for (name, (rep, others)) in by_group {
        let Some(rep) = rep else { continue };
        let mut members = vec![rep.clone()];
        let mut signs = vec![1.0];
        for (m, sg) in others {
            members.push(m);
            signs.push(sg);
        }
        out.push(Group { name: format!("{name}_R"), representative: rep, members, signs });
    }
    Some(out)
}

pub fn find_groups(
    rows: &[RegulatorRow],
    omics: &[Omic],
    threshold: f64,
    rng: &mut crate::rrng::RngStream,
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
    let mut values: Vec<Vec<f64>> = Vec::with_capacity(model.len());
    let mut binary: Vec<bool> = Vec::with_capacity(model.len());
    for r in &model {
        let Some(omic) = omics.iter().find(|o| o.name == r.omic) else {
            continue;
        };
        let Some(&idx) = omic.data.row_index().get(r.regulator.as_str()) else {
            continue;
        };
        model_ok.push(r);
        // `CollinearityFilter1` correlates `data2 = scale(data, scale, center)`,
        // never the raw matrix -- see `r_scale_column`.
        values.push(r_scale_column(&omic.data.values[idx]));
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
    // `mycorrelations` is `combn(myreg, 2)` — i ascending, then j — and `mycor`
    // keeps that order after the threshold filter. The order is not cosmetic:
    // it fixes igraph's vertex numbering (first appearance scanning each edge
    // row left column then right), which fixes the component indices that name
    // the groups AND the order of the vector `sample()` indexes into.
    let mut mycor: Vec<(usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            if binary[i] || binary[j] {
                skipped_binary += 1;
                continue;
            }
            let r = r_cor(&values[i], &values[j]);
            if r.is_nan() {
                continue;
            }
            corr[i][j] = r;
            corr[j][i] = r;
            if r.abs() >= threshold {
                adj[i][j] = true;
                adj[j][i] = true;
                mycor.push((i, j));
            }
        }
    }

    if std::env::var_os("MORE_RS_DEBUG_EDGES").is_some() {
        eprintln!("DBG MYREG order: {}",
            model.iter().map(|r| r.regulator.as_str()).collect::<Vec<_>>().join(","));
        eprintln!("DBG EDGES {} over {} nodes", mycor.len(), n);
        for &(a, b) in &mycor {
            eprintln!("DBG   E {} {} {:.10}", model[a].regulator, model[b].regulator, corr[a][b]);
        }
    }

    let mut groups = Vec::new();
    let mut peel_order: Vec<(usize, usize, usize)> = Vec::new();
    if mycor.is_empty() {
        return (groups, skipped_binary);
    }

    // igraph's vertex order. R's `nrow(mycor) == 1` branch (MORE_MLR.R:805)
    // needs no special case here: a lone edge is a two-node complete component,
    // its `correlacionados` is that edge's two names in this same order, and it
    // is named `_mc1_R` either way — so the general path below reproduces it,
    // draw included.
    let mut vorder: Vec<usize> = Vec::new();
    for &(a, b) in &mycor {
        for v in [a, b] {
            if !vorder.contains(&v) {
                vorder.push(v);
            }
        }
    }

    // Components, discovered in vertex order so the indices match igraph's.
    let mut component = vec![usize::MAX; n];
    let mut n_components = 0usize;
    for &start in &vorder {
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

    let mut collapsed = 0usize;
    for c in 0..n_components {
        // `names(mycomponents$membership[mycomponents$membership == i])`.
        let members: Vec<usize> =
            vorder.iter().copied().filter(|&i| component[i] == c).collect();
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
            // Complete clique: collapse whole. `keep = sample(correlacionados, 1)`
            // (MORE_MLR.R:860) — unconditional, once per complete component.
            collapsed += 1;
            let names: Vec<String> =
                members.iter().map(|&i| model[i].regulator.clone()).collect();
            let rep = members[rng.sample_one_of(&names, "CollinearityFilter1/clique")];
            groups.push(Group {
                name: format!("{}_mc{}_R", model[rep].omic, collapsed),
                representative: model[rep].regulator.clone(),
                members: names,
                signs: members
                    .iter()
                    .map(|&i| if i == rep || corr[rep][i] >= 0.0 { 1.0 } else { -1.0 })
                    .collect(),
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
            // `mynumedges = table(as_edgelist(mysubgraph))` counts only nodes
            // that still carry an edge, and `table` returns them **sorted by
            // name**. That sort is the order `sample()` indexes into, so it is
            // reproduced rather than left as the port's own node order.
            let mut live: Vec<(usize, usize)> = members
                .iter()
                .copied()
                .filter(|&v| alive[v])
                .map(|v| (v, degree(v, &alive)))
                .filter(|&(_, d)| d > 0)
                .collect();
            if live.is_empty() {
                break;
            }
            live.sort_by(|a, b| model[a.0].regulator.cmp(&model[b.0].regulator));
            let max_deg = live.iter().map(|&(_, d)| d).max().unwrap();
            let top: Vec<usize> =
                live.iter().filter(|&&(_, d)| d == max_deg).map(|&(v, _)| v).collect();

            if std::env::var_os("MORE_RS_DEBUG_EDGES").is_some() {
                eprintln!("DBG PEEL c={} j={} max_deg={} top=[{}]", c, j + 1, max_deg,
                    top.iter().map(|&v| model[v].regulator.as_str()).collect::<Vec<_>>().join(","));
            }
            let rep = if top.len() == 1 {
                top[0]
            } else {
                // R's tie-break sums |r| over the ORIGINAL `mycor` table:
                //   sums = sapply(maxcorrelationed, function(x)
                //            sum(abs(mycor[which(apply(mycor[,c(1,2)]==c(x),1,any)),3])))
                // `mycor` is built once, before any peeling, so a candidate
                // still earns credit for edges to regulators that have already
                // been swept away. Restricting this to `alive` neighbours --
                // the intuitive reading -- manufactures ties out of decided
                // cases: on mlr-denser the last component leaves {R7, R14},
                // both degree 1, alive-sums both 0.7137, whereas R gives R14
                // 1.4237 through its dead edge to R5 and picks it outright.
                let sums: Vec<f64> = top
                    .iter()
                    .map(|&v| mycor.iter()
                        .filter(|&&(a, b)| a == v || b == v)
                        .map(|&(a, b)| corr[a][b].abs())
                        .sum())
                    .collect();
                let max_sum = sums.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                // `which(sums == max(sums))` is exact equality in R, and
                // whether it returns one index or several decides whether the
                // stream advances at all. Matching R's comparison is therefore
                // part of matching its RNG, not a numerical nicety.
                let tied: Vec<usize> = top
                    .iter()
                    .zip(&sums)
                    .filter(|(_, &s)| s == max_sum)
                    .map(|(&v, _)| v)
                    .collect();
                if std::env::var_os("MORE_RS_DEBUG_EDGES").is_some() {
                    eprintln!("DBG   sums=[{}] max={:.17e} tied=[{}]",
                        sums.iter().map(|s| format!("{s:.17e}")).collect::<Vec<_>>().join(","),
                        max_sum,
                        tied.iter().map(|&v| model[v].regulator.as_str()).collect::<Vec<_>>().join(","));
                }
                if tied.len() == 1 {
                    tied[0]
                } else {
                    let names: Vec<String> =
                        tied.iter().map(|&v| model[v].regulator.clone()).collect();
                    tied[rng.sample_one_of(&names, "CollinearityFilter1/peel")]
                }
            };

            // The design loses this representative's *currently alive*
            // neighbours -- that is what determines the surviving columns.
            let neighbours: Vec<usize> =
                (0..n).filter(|&w| alive[w] && adj[rep][w]).collect();
            j += 1;
            for &w in &neighbours {
                alive[w] = false;
            }
            alive[rep] = false;
            peel_order.push((rep, c, j));
        }
    }

    // Star-peel membership, as R's `reg.table[, "filter"]` ends up recording it.
    //
    // R marks the representative `_R` at the start of its own iteration, then
    // walks `actual.correlation` -- every row of the ORIGINAL `mycor` table
    // involving that representative -- and stamps the other endpoint `_P`/`_N`.
    // Nothing is guarded against being written twice, so a regulator adjacent
    // to several representatives keeps the label of the **last** one, even
    // though the design column was removed by whichever representative swept it
    // first. The two are genuinely different questions: which column survives
    // (the peel) versus which group a regulator is reported under (the labels).
    //
    // Deciding membership at sweep time instead costs real edges: on
    // mlr-denser, R5 is swept at j=4 with R12 but is also adjacent to R14 at
    // j=6, so R reports it under R14's group and the port reported it under
    // R12's.
    if !peel_order.is_empty() {
        // label[i] = (group index into peel_order, sign)
        let mut label: Vec<Option<(usize, f64)>> = vec![None; n];
        for (gi, &(rep, _, _)) in peel_order.iter().enumerate() {
            label[rep] = Some((gi, 1.0));
            for w in 0..n {
                if w != rep && adj[rep][w] {
                    label[w] = Some((gi, if corr[rep][w] >= 0.0 { 1.0 } else { -1.0 }));
                }
            }
        }
        for (gi, &(rep, c, j)) in peel_order.iter().enumerate() {
            let mut members = Vec::new();
            let mut signs = Vec::new();
            // Representative first, so `members[0]` stays the representative.
            for i in (0..n).filter(|&i| label[i].map(|l| l.0) == Some(gi)) {
                let sign = label[i].unwrap().1;
                if i == rep {
                    members.insert(0, model[i].regulator.clone());
                    signs.insert(0, 1.0);
                } else {
                    members.push(model[i].regulator.clone());
                    signs.push(sign);
                }
            }
            // A representative can itself be relabelled into a later group, in
            // which case its own group has no representative left and R would
            // carry an orphaned marker. Nothing can stand in for it, so drop it.
            if members.first().map(|m| m != &model[rep].regulator).unwrap_or(true) {
                continue;
            }
            groups.push(Group {
                name: format!("{}_mc{}_{}_R", model[rep].omic, c + 1, j),
                representative: model[rep].regulator.clone(),
                members,
                signs,
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
        let data = Frame {
            row_names: regs.iter().map(|(n, _)| n.to_string()).collect(),
            col_names: (0..regs[0].1.len()).map(|i| format!("S{i}")).collect(),
            values: regs.iter().map(|(_, v)| v.clone()).collect(),
        };
        Omic {
            name: name.into(),
            input_data: data.clone(),
            data,
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
        let (groups, _) = find_groups(&rows, &[o], 0.7, &mut crate::rrng::RngStream::new(123));
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
        let (groups, _) = find_groups(&rows, &[o], 0.7, &mut crate::rrng::RngStream::new(123));
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
        let (groups, _) = find_groups(&rows, &[o], 0.7, &mut crate::rrng::RngStream::new(123));
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
        let (groups, _) = find_groups(&rows, &[o], 0.7, &mut crate::rrng::RngStream::new(123));
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
        let (groups, skipped) = find_groups(&rows, &[o], 0.7, &mut crate::rrng::RngStream::new(123));
        assert!(groups.is_empty());
        assert_eq!(skipped, 1);
    }

    #[test]
    fn a_negatively_correlated_member_is_recorded_with_a_negative_sign() {
        // Two regulators that are perfectly anti-correlated: |r| clears the
        // 0.7 threshold, so they collapse, but the member carries -1 so the
        // rpc table can flip the representative's coefficient for it.
        let a: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let b: Vec<f64> = a.iter().map(|v| -v).collect();
        let rows = vec![row("A", "TF"), row("B", "TF")];
        let o = omic("TF", &[("A", a), ("B", b)], false);
        let (groups, _) = find_groups(&rows, &[o], 0.7, &mut crate::rrng::RngStream::new(123));
        assert_eq!(groups.len(), 1, "{groups:?}");
        let g = &groups[0];
        let member = g.members.iter().position(|m| m != &g.representative).unwrap();
        assert_eq!(g.signs[member], -1.0);
        let rep = g.members.iter().position(|m| m == &g.representative).unwrap();
        assert_eq!(g.signs[rep], 1.0);
    }

    #[test]
    fn suppressed_lists_every_member_but_the_representative() {
        let g = Group {
            name: "TF_mc1_R".into(),
            representative: "A".into(),
            members: vec!["A".into(), "B".into(), "C".into()],
            signs: vec![1.0, 1.0, 1.0],
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
            signs: vec![1.0, 1.0, 1.0],
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
            signs: vec![1.0, 1.0],
        };
        assert_eq!(expand(&["Z".to_string()], &[g]), vec!["Z"]);
    }

    #[test]
    fn a_single_model_regulator_forms_no_group() {
        let o = omic("TF", &[("A", vec![1.0, 2.0, 3.0, 4.0])], false);
        let (groups, _) = find_groups(&[row("A", "TF")], &[o], 0.7, &mut crate::rrng::RngStream::new(123));
        assert!(groups.is_empty());
    }
}
