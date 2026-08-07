//! Equivalence test against the R reference stack.
//!
//! The fixture is produced by `equivalence/pls_oracle.R`, which runs
//! `ropls::opls` with exactly the arguments MORE uses and then MORE's own
//! `p.valuejack` over the same design. Regenerate with:
//!
//! ```text
//! cd rust && Rscript equivalence/pls_oracle.R
//! ```
//!
//! The fixture carries the already-scaled X and y at full precision, so this
//! test isolates the PLS algorithm: a disagreement here is the port's fault,
//! not a difference in how the two languages centre a column.
//!
//! Tolerances are measured, not assumed — `report_tolerances` prints the worst
//! observed deviation for each quantity, which is what the port's equivalence
//! claim is stated in terms of.

use crate::jackknife;
use crate::matrix::Mat;
use crate::pls;

const FIXTURE: &str = include_str!("../equivalence/fixtures/pls_oracle.tsv");

struct Oracle {
    n: usize,
    p: usize,
    crossval: usize,
    x: Mat,
    y: Vec<f64>,
    ncomp: usize,
    coefficients: Vec<f64>,
    vip: Vec<f64>,
    r2y_cum: f64,
    q2_cum: f64,
    rmsee: f64,
    fitted: Vec<f64>,
    pvalue: Vec<f64>,
}

fn field<'a>(key: &str) -> &'a str {
    FIXTURE
        .lines()
        .find(|l| l.split('\t').next() == Some(key))
        .unwrap_or_else(|| panic!("fixture has no '{key}' row"))
        .splitn(2, '\t')
        .nth(1)
        .unwrap_or_else(|| panic!("fixture row '{key}' has no value"))
}

fn nums(key: &str) -> Vec<f64> {
    field(key)
        .split('\t')
        .map(|v| v.parse::<f64>().expect("fixture value is not a number"))
        .collect()
}

fn one(key: &str) -> f64 {
    nums(key)[0]
}

fn load() -> Oracle {
    let n = one("n") as usize;
    let p = one("p") as usize;
    // R writes a matrix column-major, which is this crate's layout too.
    let flat = nums("x");
    assert_eq!(flat.len(), n * p, "fixture x has the wrong element count");
    let cols: Vec<Vec<f64>> = (0..p).map(|j| flat[j * n..(j + 1) * n].to_vec()).collect();

    Oracle {
        n,
        p,
        crossval: one("crossval") as usize,
        x: Mat::from_columns(&cols),
        y: nums("y"),
        ncomp: one("ncomp") as usize,
        coefficients: nums("coefficients"),
        vip: nums("vip"),
        r2y_cum: one("r2y_cum"),
        q2_cum: one("q2_cum"),
        rmsee: one("rmsee"),
        fitted: nums("fitted"),
        pvalue: nums("pvalue"),
    }
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len(), "length mismatch against the oracle");
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max)
}

/// Deviations above this would still be far too small to move a significance
/// decision, but tight enough that a genuine algorithmic divergence trips it.
const TOL: f64 = 1e-9;

#[test]
fn the_component_count_matches_ropls() {
    let o = load();
    let fit = pls::fit(&o.x, &o.y, None, o.crossval).expect("port produced no model");
    assert_eq!(
        fit.n_comp, o.ncomp,
        "component count diverged; every coefficient depends on this"
    );
}

#[test]
fn coefficients_match_ropls() {
    let o = load();
    let fit = pls::fit(&o.x, &o.y, None, o.crossval).expect("port produced no model");
    let d = max_abs_diff(&fit.coefficients, &o.coefficients);
    assert!(d < TOL, "max coefficient deviation {d:e} exceeds {TOL:e}");
}

#[test]
fn vip_matches_ropls() {
    let o = load();
    let fit = pls::fit(&o.x, &o.y, None, o.crossval).expect("port produced no model");
    let d = max_abs_diff(&fit.vip, &o.vip);
    assert!(d < TOL, "max VIP deviation {d:e} exceeds {TOL:e}");
}

#[test]
fn fitted_values_match_ropls() {
    let o = load();
    let fit = pls::fit(&o.x, &o.y, None, o.crossval).expect("port produced no model");
    let d = max_abs_diff(&fit.fitted, &o.fitted);
    assert!(d < TOL, "max fitted deviation {d:e} exceeds {TOL:e}");
}

#[test]
fn goodness_of_fit_matches_ropls_including_its_rounding() {
    let o = load();
    let fit = pls::fit(&o.x, &o.y, None, o.crossval).expect("port produced no model");
    // These are compared exactly: both sides are signif(x, 3), so any
    // difference means the port rounded differently or computed a different
    // number, and MORE writes R2Y(cum) into the rpc table.
    assert_eq!(fit.r2y_cum, o.r2y_cum, "R2Y(cum)");
    assert_eq!(fit.q2_cum, o.q2_cum, "Q2(cum)");
    assert_eq!(fit.rmsee, o.rmsee, "RMSEE");
}

#[test]
fn jackknife_p_values_match_more() {
    let o = load();
    let fit = pls::fit(&o.x, &o.y, None, o.crossval).expect("port produced no model");
    let p = jackknife::p_values(&o.x, &o.y, &fit);
    let d = max_abs_diff(&p, &o.pvalue);
    assert!(d < TOL, "max p-value deviation {d:e} exceeds {TOL:e}");
}

#[test]
fn the_significance_decision_is_identical() {
    // The decision, not the numbers, is what the goal calls a hard failure.
    let o = load();
    let fit = pls::fit(&o.x, &o.y, None, o.crossval).expect("port produced no model");
    let p = jackknife::p_values(&o.x, &o.y, &fit);

    let ours: Vec<usize> = (0..o.p)
        .filter(|&j| fit.vip[j] > 0.8 && p[j] < 0.05)
        .collect();
    let theirs: Vec<usize> = (0..o.p)
        .filter(|&j| o.vip[j] > 0.8 && o.pvalue[j] < 0.05)
        .collect();
    assert_eq!(ours, theirs, "selected variable sets differ");
    assert!(!theirs.is_empty(), "fixture selects nothing, so it tests nothing");
}

/// Not an assertion — prints the measured deviations so the equivalence claim
/// can quote real numbers. Run with `cargo test -- --nocapture report_tolerances`.
#[test]
fn report_tolerances() {
    let o = load();
    let fit = pls::fit(&o.x, &o.y, None, o.crossval).expect("port produced no model");
    let p = jackknife::p_values(&o.x, &o.y, &fit);
    println!("n={} p={} ncomp={}", o.n, o.p, fit.n_comp);
    println!("  coefficients  max|d| = {:e}", max_abs_diff(&fit.coefficients, &o.coefficients));
    println!("  vip           max|d| = {:e}", max_abs_diff(&fit.vip, &o.vip));
    println!("  fitted        max|d| = {:e}", max_abs_diff(&fit.fitted, &o.fitted));
    println!("  p-values      max|d| = {:e}", max_abs_diff(&p, &o.pvalue));
}
