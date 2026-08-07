//! Elastic net by coordinate descent, with cross-validated alpha and lambda.
//!
//! Ports `ElasticNet` (`../R/auxFunctions.R`) as `runMORE.R` reaches it: the
//! script never passes `alfaEN`, so `more()`'s default `NULL` selects the
//! branch that fits **eleven** `cv.glmnet` models per target — `alphas =
//! seq(0, 1, 0.1)` — and keeps the alpha whose `cvup` at `lambda.min` is
//! smallest.
//!
//! # The one deliberate divergence, and its measured cost
//!
//! `cv.glmnet` draws its folds from R's Mersenne-Twister, seeded once by
//! `set.seed(123)` in `more()` and advanced sequentially across targets. This
//! port uses **deterministic interleaved folds** instead, the same scheme ropls
//! uses on the PLS1 path.
//!
//! That is a real difference with a named mechanism — fold assignment — not a
//! numerical artefact, and it is bounded by a measurement rather than a guess.
//! R disagrees with *itself* by this much when only the seed changes
//! (12 targets x 12 regulators x 20 samples, `equivalence/mlr_seed_spread.R`):
//!
//! ```text
//! seed 123 twice     identical
//! seed 123 vs 456    72 vs 80 edges, symmetric difference 12, Jaccard 0.854
//! ```
//!
//! So ~15% of MLR's edge set is seed-dependent inside R itself. The port is
//! therefore held to the brief's criterion for stochastic paths — precision and
//! recall inside R's own seed-to-seed spread — and not to set-equality, which
//! no implementation can reach here without reproducing R's RNG stream
//! bit-for-bit. See `SPEC.md` §4.
//!
//! Reproducing R's RNG remains the route to set-equality if that is ever
//! wanted; nothing here forecloses it, since the fold assignment is a single
//! function.

use crate::matrix::{dot, Mat};

/// Coefficients of a fitted elastic net, excluding the intercept.
#[derive(Clone, Debug)]
pub struct EnFit {
    pub intercept: f64,
    pub coefficients: Vec<f64>,
    pub alpha: f64,
    pub lambda: f64,
    /// Fraction of null deviance explained — glmnet's `dev.ratio`, which
    /// `modelcharac` rounds to 6 digits and reports as R.squared.
    pub dev_ratio: f64,
}

/// Soft-thresholding operator.
#[inline]
fn soft(z: f64, g: f64) -> f64 {
    if z > g {
        z - g
    } else if z < -g {
        z + g
    } else {
        0.0
    }
}

/// Coordinate descent for a single (alpha, lambda), warm-started from `beta`.
///
/// Objective, matching glmnet with `standardize = FALSE` and a gaussian family:
/// `1/(2n)||y - b0 - Xb||^2 + lambda*(alpha*|b|_1 + (1-alpha)/2*|b|^2)`.
fn descend(
    x: &Mat,
    y: &[f64],
    alpha: f64,
    lambda: f64,
    beta: &mut [f64],
    xx: &[f64],
    thresh: f64,
    max_iter: usize,
) {
    let n = x.nrow() as f64;
    let p = x.ncol();
    // Residual, recomputed once then updated incrementally.
    let mut r: Vec<f64> = y.to_vec();
    for j in 0..p {
        if beta[j] != 0.0 {
            for (ri, xv) in r.iter_mut().zip(x.col(j)) {
                *ri -= beta[j] * xv;
            }
        }
    }

    let l1 = lambda * alpha;
    let l2 = lambda * (1.0 - alpha);

    for _ in 0..max_iter {
        let mut max_change = 0.0f64;
        for j in 0..p {
            if xx[j] == 0.0 {
                continue;
            }
            let old = beta[j];
            // Partial residual correlation for coordinate j.
            let rho = dot(x.col(j), &r) / n + (xx[j] / n) * old;
            let new = soft(rho, l1) / (xx[j] / n + l2);
            if new != old {
                let delta = new - old;
                for (ri, xv) in r.iter_mut().zip(x.col(j)) {
                    *ri -= delta * xv;
                }
                beta[j] = new;
                max_change = max_change.max(delta.abs() * (xx[j] / n).sqrt());
            }
        }
        if max_change < thresh {
            break;
        }
    }
}

/// glmnet's lambda sequence: geometric from `lambda_max` down to
/// `lambda_min_ratio * lambda_max`, `nlambda` points.
fn lambda_path(x: &Mat, y: &[f64], alpha: f64, nlambda: usize) -> Vec<f64> {
    let n = x.nrow() as f64;
    // glmnet substitutes a small alpha when computing lambda_max for ridge,
    // otherwise lambda_max would be infinite.
    let a = alpha.max(1e-3);
    let mut lmax: f64 = 0.0;
    for j in 0..x.ncol() {
        lmax = lmax.max((dot(x.col(j), y) / n).abs());
    }
    lmax /= a;
    if !(lmax > 0.0) {
        return vec![0.0];
    }
    let ratio = if x.nrow() < x.ncol() { 0.01 } else { 1e-4 };
    let lmin = lmax * ratio;
    let step = (lmin / lmax).ln() / (nlambda as f64 - 1.0);
    (0..nlambda).map(|i| lmax * (step * i as f64).exp()).collect()
}

/// Fit the whole lambda path at one alpha, warm-starting down the path.
fn path_fit(x: &Mat, y: &[f64], alpha: f64, lambdas: &[f64], thresh: f64) -> Vec<Vec<f64>> {
    let p = x.ncol();
    let xx: Vec<f64> = (0..p).map(|j| dot(x.col(j), x.col(j))).collect();
    let mut beta = vec![0.0; p];
    let mut out = Vec::with_capacity(lambdas.len());
    for &l in lambdas {
        descend(x, y, alpha, l, &mut beta, &xx, thresh, 1000);
        out.push(beta.clone());
    }
    out
}

/// Deterministic interleaved folds — see the module note on why these are not
/// drawn at random.
fn folds(n: usize, k: usize) -> Vec<Vec<usize>> {
    let k = k.clamp(1, n.max(1));
    let mut f = vec![Vec::new(); k];
    for i in 0..n {
        f[i % k].push(i);
    }
    f.retain(|v| !v.is_empty());
    f
}

/// `mynfolds` from `ElasticNet`: leave-one-out below 50 observations, then
/// 5 / 7 / 10 as the sample count grows.
fn n_folds(n: usize) -> usize {
    if n < 50 {
        n
    } else if n < 100 {
        5
    } else if n < 200 {
        7
    } else {
        10
    }
}

/// Centre `y` and each column of `x`, returning the means. MLR fits an
/// intercept, which coordinate descent handles by centring rather than by
/// carrying an unpenalised column.
fn center(x: &Mat, y: &[f64]) -> (Mat, Vec<f64>, Vec<f64>, f64) {
    let n = x.nrow() as f64;
    let ymean = y.iter().sum::<f64>() / n;
    let yc: Vec<f64> = y.iter().map(|v| v - ymean).collect();
    let mut cols = Vec::with_capacity(x.ncol());
    let mut means = Vec::with_capacity(x.ncol());
    for j in 0..x.ncol() {
        let m = x.col(j).iter().sum::<f64>() / n;
        means.push(m);
        cols.push(x.col(j).iter().map(|v| v - m).collect::<Vec<f64>>());
    }
    (Mat::from_columns(&cols), yc, means, ymean)
}

/// Cross-validated elastic net over `alphas`, choosing the alpha with the
/// smallest `cvm + cvsd` at its own `lambda.min` — R's `cvup` rule.
pub fn cv_fit(x: &Mat, y: &[f64], alphas: &[f64], thresh: f64) -> Option<EnFit> {
    let n = x.nrow();
    let p = x.ncol();
    if n < 3 || p == 0 {
        return None;
    }
    let (xc, yc, xmeans, ymean) = center(x, y);
    let k = n_folds(n);
    let fold_sets = folds(n, k);

    let mut best: Option<(f64, EnFit)> = None; // (cvup, fit)

    for &alpha in alphas {
        let lambdas = lambda_path(&xc, &yc, alpha, 100);
        // Per-fold squared errors for every lambda.
        let mut sq: Vec<Vec<f64>> = vec![Vec::new(); lambdas.len()];
        for held in &fold_sets {
            let keep: Vec<usize> = (0..n).filter(|i| !held.contains(i)).collect();
            if keep.len() < 2 {
                continue;
            }
            let xt = xc.select_rows(&keep);
            let yt: Vec<f64> = keep.iter().map(|&i| yc[i]).collect();
            // Re-centre within the fold, as glmnet does.
            let (xt, yt, tm, tym) = center(&xt, &yt);
            let betas = path_fit(&xt, &yt, alpha, &lambdas, thresh);
            for (li, b) in betas.iter().enumerate() {
                for &i in held {
                    let mut pred = tym;
                    for j in 0..p {
                        if b[j] != 0.0 {
                            pred += b[j] * (xc.get(i, j) - tm[j]);
                        }
                    }
                    let e = yc[i] - pred;
                    sq[li].push(e * e);
                }
            }
        }

        // cvm and its standard error per lambda, then lambda.min and cvup.
        let mut best_li = 0usize;
        let mut best_cvm = f64::INFINITY;
        let mut best_cvup = f64::INFINITY;
        for (li, errs) in sq.iter().enumerate() {
            if errs.is_empty() {
                continue;
            }
            let m = errs.iter().sum::<f64>() / errs.len() as f64;
            if m < best_cvm {
                let var = errs.iter().map(|e| (e - m) * (e - m)).sum::<f64>()
                    / (errs.len().max(2) as f64 - 1.0);
                best_cvm = m;
                best_cvup = m + (var / errs.len() as f64).sqrt();
                best_li = li;
            }
        }
        if !best_cvm.is_finite() {
            continue;
        }

        let full = path_fit(&xc, &yc, alpha, &lambdas, thresh);
        let beta = full[best_li].clone();

        // dev.ratio = 1 - RSS/null deviance.
        let mut rss = 0.0;
        let mut tss = 0.0;
        for i in 0..n {
            let mut pred = 0.0;
            for j in 0..p {
                if beta[j] != 0.0 {
                    pred += beta[j] * xc.get(i, j);
                }
            }
            rss += (yc[i] - pred) * (yc[i] - pred);
            tss += yc[i] * yc[i];
        }
        let dev_ratio = if tss > 0.0 { 1.0 - rss / tss } else { 0.0 };

        let intercept = ymean - (0..p).map(|j| beta[j] * xmeans[j]).sum::<f64>();
        let fit = EnFit {
            intercept,
            coefficients: beta,
            alpha,
            lambda: lambdas[best_li],
            dev_ratio,
        };
        if best.as_ref().map_or(true, |(c, _)| best_cvup < *c) {
            best = Some((best_cvup, fit));
        }
    }

    best.map(|(_, f)| f)
}

/// The alpha grid `ElasticNet` uses when `alfaEN` is NULL — which is always,
/// from `runMORE.R`.
pub fn default_alphas() -> Vec<f64> {
    (0..=10).map(|i| i as f64 / 10.0).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn design(n: usize) -> (Mat, Vec<f64>) {
        let x1: Vec<f64> = (0..n).map(|i| ((i * 7 % 23) as f64) / 23.0 - 0.5).collect();
        let x2: Vec<f64> = (0..n).map(|i| ((i * 13 % 19) as f64) / 19.0 - 0.5).collect();
        let x3: Vec<f64> = (0..n).map(|i| ((i * 5 % 17) as f64) / 17.0 - 0.5).collect();
        let y: Vec<f64> = x1.iter().map(|v| 4.0 * v).collect();
        (Mat::from_columns(&[x1, x2, x3]), y)
    }

    #[test]
    fn folds_are_interleaved_and_cover_every_observation() {
        let f = folds(7, 3);
        let mut all: Vec<usize> = f.iter().flatten().copied().collect();
        all.sort();
        assert_eq!(all, (0..7).collect::<Vec<_>>());
        assert_eq!(f[0], vec![0, 3, 6]);
    }

    #[test]
    fn fold_count_follows_the_r_rule() {
        assert_eq!(n_folds(20), 20);
        assert_eq!(n_folds(60), 5);
        assert_eq!(n_folds(150), 7);
        assert_eq!(n_folds(500), 10);
    }

    #[test]
    fn soft_thresholding_shrinks_towards_zero() {
        assert_eq!(soft(0.5, 0.2), 0.3);
        assert_eq!(soft(-0.5, 0.2), -0.3);
        assert_eq!(soft(0.1, 0.2), 0.0);
    }

    #[test]
    fn a_large_lambda_zeroes_every_coefficient() {
        let (x, y) = design(20);
        let (xc, yc, _, _) = center(&x, &y);
        let xx: Vec<f64> = (0..xc.ncol()).map(|j| dot(xc.col(j), xc.col(j))).collect();
        let mut beta = vec![0.0; xc.ncol()];
        descend(&xc, &yc, 1.0, 1e6, &mut beta, &xx, 1e-7, 100);
        assert!(beta.iter().all(|b| *b == 0.0));
    }

    #[test]
    fn lambda_max_is_the_smallest_lambda_that_zeroes_everything() {
        let (x, y) = design(20);
        let (xc, yc, _, _) = center(&x, &y);
        let path = lambda_path(&xc, &yc, 1.0, 100);
        let xx: Vec<f64> = (0..xc.ncol()).map(|j| dot(xc.col(j), xc.col(j))).collect();
        let mut beta = vec![0.0; xc.ncol()];
        descend(&xc, &yc, 1.0, path[0], &mut beta, &xx, 1e-7, 100);
        assert!(beta.iter().all(|b| b.abs() < 1e-12), "{beta:?}");
    }

    #[test]
    fn the_path_descends_from_lambda_max() {
        let (x, y) = design(20);
        let (xc, yc, _, _) = center(&x, &y);
        let path = lambda_path(&xc, &yc, 1.0, 100);
        assert_eq!(path.len(), 100);
        assert!(path[0] > path[99]);
    }

    #[test]
    fn cross_validation_selects_the_driving_predictor() {
        let (x, y) = design(24);
        let fit = cv_fit(&x, &y, &default_alphas(), 1e-7).expect("fit");
        assert!(fit.coefficients[0].abs() > 1e-6, "driver not selected: {:?}", fit.coefficients);
    }

    #[test]
    fn an_unrelated_predictor_is_shrunk_far_below_the_driver() {
        let (x, y) = design(24);
        let fit = cv_fit(&x, &y, &default_alphas(), 1e-7).expect("fit");
        assert!(fit.coefficients[0].abs() > 10.0 * fit.coefficients[1].abs());
    }

    #[test]
    fn a_good_fit_reports_a_high_dev_ratio() {
        let (x, y) = design(24);
        let fit = cv_fit(&x, &y, &default_alphas(), 1e-7).expect("fit");
        assert!(fit.dev_ratio > 0.9, "dev_ratio was {}", fit.dev_ratio);
    }

    #[test]
    fn the_intercept_recovers_the_response_mean_offset() {
        let (x, mut y) = design(24);
        for v in y.iter_mut() {
            *v += 7.0;
        }
        let fit = cv_fit(&x, &y, &default_alphas(), 1e-7).expect("fit");
        assert!((fit.intercept - 7.0).abs() < 0.5, "intercept {}", fit.intercept);
    }

    #[test]
    fn the_default_alpha_grid_is_zero_to_one_by_tenths() {
        let a = default_alphas();
        assert_eq!(a.len(), 11);
        assert_eq!(a[0], 0.0);
        assert_eq!(a[10], 1.0);
    }

    #[test]
    fn refitting_the_same_data_gives_the_same_answer() {
        // Deterministic folds mean no run-to-run variation at all, unlike R.
        let (x, y) = design(24);
        let a = cv_fit(&x, &y, &default_alphas(), 1e-7).unwrap();
        let b = cv_fit(&x, &y, &default_alphas(), 1e-7).unwrap();
        assert_eq!(a.coefficients, b.coefficients);
        assert_eq!(a.lambda, b.lambda);
    }
}
