//! Elastic net by coordinate descent, with cross-validated alpha and lambda.
//!
//! Ports `ElasticNet` (`../R/auxFunctions.R`) as `runMORE.R` reaches it: the
//! script never passes `alfaEN`, so `more()`'s default `NULL` selects the
//! branch that fits **eleven** `cv.glmnet` models per target — `alphas =
//! seq(0, 1, 0.1)` — and keeps the alpha whose `cvup` at `lambda.min` is
//! smallest.
//!
//! # Fold assignment is not a divergence at MORE's sizes
//!
//! `cv.glmnet` draws its folds with `sample()`, but MORE's own rule
//! (`mynfolds`) is **leave-one-out below 50 observations**, and with
//! `nfolds == n` the draw only relabels folds that each hold one observation.
//! The partition — and therefore every number downstream — is identical
//! whatever the seed. This port uses deterministic interleaved folds, which
//! coincide exactly with R's in that regime.
//!
//! Above 50 observations R takes 5/7/10 folds and the draw does start to
//! matter; that case is not covered by the equivalence harness and the port
//! would need R's RNG stream to match it. `folds` is a single function, so
//! nothing here forecloses that.
//!
//! # What is left
//!
//! Against real `cv.glmnet` on byte-identical design matrices the port picks
//! the same winning alpha and the same selected-variable count on 11 of 12
//! mlr-denser targets. The twelfth is a near-tie: R separates alpha 0.2 from
//! 0.3 by 0.00036 in `cvup` (0.275506 vs 0.275866) and the port separates
//! them by 0.00166 the other way. Tightening glmnet's own tolerance narrows
//! R's margin to 4e-6, so the data does not determine that choice — see
//! `SPEC.md` §4.12.

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
///
/// glmnet's Fortran solves this on `y` divided by its population standard
/// deviation `ys`, then multiplies both the coefficients and the reported
/// lambdas by `ys` on the way out. Undoing that rescaling is *not* symmetric
/// between the two penalties:
///
/// ```text
/// b_k = ys * soft(rho/ys, lambda_int*alpha) / (xv_k + lambda_int*(1-alpha))
///     =      soft(rho,    lambda_R*alpha)   / (xv_k + lambda_R*(1-alpha)/ys)
/// ```
///
/// The L1 threshold comes out unchanged, so `lambda_max` agrees with glmnet
/// to every printed digit even when the ridge term is wrong — which is
/// exactly how a factor-of-`ys` error in the L2 term hid here. Pass `ys`, not
/// 1.0, or the solver converges cleanly to the wrong optimum and tightening
/// `thresh` moves the answer *away* from R.
///
/// `thresh` is likewise the already-rescaled tolerance: glmnet tests
/// `max_j xv_j*delta_j^2 < thresh` in its internal units, so the caller
/// multiplies by `ys^2`. The criterion is quadratic in `delta`; a
/// square-rooted version converges far tighter than glmnet and shifts where
/// the path stops.
/// glmnet switches to naive (residual) updates at this many variables; below
/// it, `type.gaussian = "covariance"` is the default and the gradient is
/// carried through the Gram matrix instead.
const COVARIANCE_MAX_VARS: usize = 500;

fn descend(
    x: &Mat,
    y: &[f64],
    alpha: f64,
    lambda: f64,
    ys: f64,
    beta: &mut [f64],
    xx: &[f64],
    thresh: f64,
    max_iter: usize,
) {
    let n = x.nrow() as f64;
    let p = x.ncol();

    let l1 = lambda * alpha;
    let l2 = lambda * (1.0 - alpha) / ys;

    // Residual for the current (possibly warm-started) beta.
    let mut r: Vec<f64> = y.to_vec();
    for j in 0..p {
        if beta[j] != 0.0 {
            for (ri, xv) in r.iter_mut().zip(x.col(j)) {
                *ri -= beta[j] * xv;
            }
        }
    }

    // Covariance mode carries `g[j] = <x_j, r>` forward by subtracting
    // `delta * <x_j, x_k>` on every update, exactly as glmnet's `elnet1` does,
    // instead of recomputing the inner product from a maintained residual each
    // sweep. The two are algebraically identical and numerically are not: the
    // recomputed form is the more accurate one, but it converges to a slightly
    // different iterate inside the same tolerance ball, and MORE runs glmnet at
    // `epsilon = 1e-5` where that ball is wide enough to change which alpha wins
    // a cross-validation tie. Gram columns are built lazily, so only variables
    // that actually enter the model cost anything.
    let covariance = p < COVARIANCE_MAX_VARS;
    let mut g: Vec<f64> = if covariance {
        (0..p).map(|j| dot(x.col(j), &r)).collect()
    } else {
        Vec::new()
    };
    let mut gram: Vec<Option<Vec<f64>>> = if covariance { vec![None; p] } else { Vec::new() };

    // glmnet's active set is every variable that has *ever* been nonzero on
    // this path, not the currently nonzero ones — entries are never dropped.
    let mut ever_active: Vec<bool> = beta.iter().map(|b| *b != 0.0).collect();

    // One Gauss-Seidel sweep. Returns glmnet's `dlx`.
    let mut sweep = |beta: &mut [f64],
                     r: &mut Vec<f64>,
                     g: &mut Vec<f64>,
                     gram: &mut Vec<Option<Vec<f64>>>,
                     ever_active: &mut Vec<bool>,
                     active_only: bool| -> f64 {
        let mut dlx = 0.0f64;
        for j in 0..p {
            if xx[j] == 0.0 || (active_only && !ever_active[j]) {
                continue;
            }
            let old = beta[j];
            // Partial residual correlation for coordinate j.
            let rho = if covariance {
                g[j] / n + (xx[j] / n) * old
            } else {
                dot(x.col(j), r) / n + (xx[j] / n) * old
            };
            let new = soft(rho, l1) / (xx[j] / n + l2);
            if new != old {
                let delta = new - old;
                if covariance {
                    if gram[j].is_none() {
                        gram[j] = Some((0..p).map(|k| dot(x.col(k), x.col(j))).collect());
                    }
                    let col = gram[j].as_ref().unwrap();
                    for k in 0..p {
                        g[k] -= delta * col[k];
                    }
                } else {
                    for (ri, xv) in r.iter_mut().zip(x.col(j)) {
                        *ri -= delta * xv;
                    }
                }
                beta[j] = new;
                if new != 0.0 {
                    ever_active[j] = true;
                }
                dlx = dlx.max((xx[j] / n) * delta * delta);
            }
        }
        dlx
    };

    // The Fortran's two-level schedule: a full sweep, and if it did not
    // already converge, active-set sweeps to convergence before the next full
    // sweep. Exit is always off a *full* sweep.
    let mut passes = 0usize;
    while passes < max_iter {
        passes += 1;
        if sweep(beta, &mut r, &mut g, &mut gram, &mut ever_active, false) < thresh {
            break;
        }
        while passes < max_iter {
            passes += 1;
            if sweep(beta, &mut r, &mut g, &mut gram, &mut ever_active, true) < thresh {
                break;
            }
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

/// `glmnet.control()` defaults that govern where the path stops.
const FDEV: f64 = 1e-5;
const DEVMAX: f64 = 0.999;
const MNLAM: usize = 5;

/// Fraction of null deviance explained by `beta` on already-centred data —
/// glmnet's `rsq` / `dev.ratio`.
fn dev_ratio(x: &Mat, y: &[f64], beta: &[f64], tss: f64) -> f64 {
    if !(tss > 0.0) {
        return 0.0;
    }
    let mut rss = 0.0;
    for i in 0..x.nrow() {
        let mut pred = 0.0;
        for (j, &b) in beta.iter().enumerate() {
            if b != 0.0 {
                pred += b * x.get(i, j);
            }
        }
        rss += (y[i] - pred) * (y[i] - pred);
    }
    1.0 - rss / tss
}

/// Fit the lambda path at one alpha, warm-starting down the path.
///
/// `truncate` reproduces the Fortran kernel's early exit, which is *not*
/// cosmetic: glmnet stops adding lambdas once the fit stops improving
/// (`rsq - rsq0 < fdev*rsq`) or has essentially saturated (`rsq > devmax`),
/// having always emitted at least `mnlam` of them. A 20x17 design typically
/// yields 50-60 lambdas, not 100 — so the small, near-interpolating end of
/// the grid is never offered to cross-validation at all. Evaluating it
/// anyway lets CV pick a lambda R could not have picked, which is how the
/// port used to report R2 = 0.999 on targets where R reports 0.049.
///
/// The rule is suppressed (`flmin >= 1` in the Fortran) whenever the caller
/// supplies the lambda vector, which is exactly the per-fold case inside
/// `cv.glmnet`: every fold fits the full sequence the outer fit produced.
fn path_fit(
    x: &Mat,
    y: &[f64],
    alpha: f64,
    lambdas: &[f64],
    thresh: f64,
    truncate: bool,
) -> Vec<Vec<f64>> {
    let p = x.ncol();
    let xx: Vec<f64> = (0..p).map(|j| dot(x.col(j), x.col(j))).collect();
    let tss: f64 = y.iter().map(|v| v * v).sum();
    // glmnet's internal y-standardisation: `y` is already centred here, so
    // this is the population standard deviation it divides through by.
    let ys = (tss / x.nrow() as f64).sqrt();
    let thresh = thresh * ys * ys;
    let mut beta = vec![0.0; p];
    let mut out = Vec::with_capacity(lambdas.len());
    let mut rsq0 = 0.0f64;
    let mnl = MNLAM.min(lambdas.len());
    for (m, &l) in lambdas.iter().enumerate() {
        // glmnet substitutes an effectively infinite first lambda when it
        // generates the path itself, so the leading solution is exactly zero
        // even for ridge. With a supplied path it uses the value as given.
        if truncate && m == 0 {
            out.push(beta.clone());
            continue;
        }
        descend(x, y, alpha, l, ys, &mut beta, &xx, thresh, 100_000);
        out.push(beta.clone());
        if !truncate {
            continue;
        }
        let rsq = dev_ratio(x, y, &beta, tss);
        if m + 1 >= mnl {
            if rsq - rsq0 < FDEV * rsq || rsq > DEVMAX {
                break;
            }
        }
        rsq0 = rsq;
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
        // `cv.glmnet` fits the full data once to obtain the lambda sequence,
        // then hands that exact sequence to every fold. The outer fit is the
        // only one allowed to stop early, so it also fixes how many rungs
        // cross-validation ever sees.
        let grid = lambda_path(&xc, &yc, alpha, 100);
        let full = path_fit(&xc, &yc, alpha, &grid, thresh, true);
        let lambdas = &grid[..full.len()];
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
            let betas = path_fit(&xt, &yt, alpha, lambdas, thresh, false);
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

        let beta = full[best_li].clone();
        let tss: f64 = yc.iter().map(|v| v * v).sum();
        let dev_ratio = dev_ratio(&xc, &yc, &beta, tss);

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

/// The full-data path at one alpha, as `glmnet()` itself would report it:
/// `(lambda, dev.ratio, df)` per rung, already truncated. Lets the truncation
/// rule be measured without cross-validation in the way.
pub fn path_probe(x: &Mat, y: &[f64], alpha: f64, thresh: f64) -> Vec<(f64, f64, usize)> {
    let (xc, yc, _, _) = center(x, y);
    let grid = lambda_path(&xc, &yc, alpha, 100);
    let betas = path_fit(&xc, &yc, alpha, &grid, thresh, true);
    let tss: f64 = yc.iter().map(|v| v * v).sum();
    betas
        .iter()
        .enumerate()
        .map(|(i, b)| {
            (
                grid[i],
                dev_ratio(&xc, &yc, b, tss),
                b.iter().filter(|v| **v != 0.0).count(),
            )
        })
        .collect()
}

/// One alpha's cross-validation summary, in the same columns
/// `equivalence/en_probe.R` prints from real `cv.glmnet`.
#[derive(Debug)]
pub struct AlphaDiag {
    pub alpha: f64,
    pub nlam: usize,
    pub lmax: f64,
    pub lmin_path: f64,
    pub lambda_min: f64,
    pub cvm: f64,
    pub cvup: f64,
    pub nonzero: usize,
}

/// Per-alpha diagnostics for the equivalence probe. Deliberately a thin
/// re-run of `cv_fit`'s loop rather than a refactor of it: the probe must not
/// be able to drift away from the code it is measuring, and the alternative —
/// threading an optional collector through `cv_fit` — puts test scaffolding
/// on the hot path that every target pays for.
pub fn cv_probe(x: &Mat, y: &[f64], alphas: &[f64], thresh: f64) -> Vec<AlphaDiag> {
    let n = x.nrow();
    let p = x.ncol();
    let mut out = Vec::new();
    if n < 3 || p == 0 {
        return out;
    }
    let (xc, yc, _, _) = center(x, y);
    let fold_sets = folds(n, n_folds(n));

    for &alpha in alphas {
        let grid = lambda_path(&xc, &yc, alpha, 100);
        let full = path_fit(&xc, &yc, alpha, &grid, thresh, true);
        let lambdas = &grid[..full.len()];
        let mut sq: Vec<Vec<f64>> = vec![Vec::new(); lambdas.len()];
        for held in &fold_sets {
            let keep: Vec<usize> = (0..n).filter(|i| !held.contains(i)).collect();
            if keep.len() < 2 {
                continue;
            }
            let xt = xc.select_rows(&keep);
            let yt: Vec<f64> = keep.iter().map(|&i| yc[i]).collect();
            let (xt, yt, tm, tym) = center(&xt, &yt);
            for (li, b) in path_fit(&xt, &yt, alpha, lambdas, thresh, false)
                .iter()
                .enumerate()
            {
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
        let (mut li, mut cvm, mut cvup) = (0usize, f64::INFINITY, f64::INFINITY);
        for (i, errs) in sq.iter().enumerate() {
            if errs.is_empty() {
                continue;
            }
            let m = errs.iter().sum::<f64>() / errs.len() as f64;
            if m < cvm {
                let var = errs.iter().map(|e| (e - m) * (e - m)).sum::<f64>()
                    / (errs.len().max(2) as f64 - 1.0);
                cvm = m;
                cvup = m + (var / errs.len() as f64).sqrt();
                li = i;
            }
        }
        if !cvm.is_finite() {
            continue;
        }
        out.push(AlphaDiag {
            alpha,
            nlam: lambdas.len(),
            lmax: lambdas[0],
            lmin_path: lambdas[lambdas.len() - 1],
            lambda_min: lambdas[li],
            cvm,
            cvup,
            nonzero: full[li].iter().filter(|b| **b != 0.0).count(),
        });
    }
    out
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
        descend(&xc, &yc, 1.0, 1e6, 1.0, &mut beta, &xx, 1e-7, 100);
        assert!(beta.iter().all(|b| *b == 0.0));
    }

    #[test]
    fn lambda_max_is_the_smallest_lambda_that_zeroes_everything() {
        let (x, y) = design(20);
        let (xc, yc, _, _) = center(&x, &y);
        let path = lambda_path(&xc, &yc, 1.0, 100);
        let xx: Vec<f64> = (0..xc.ncol()).map(|j| dot(xc.col(j), xc.col(j))).collect();
        let mut beta = vec![0.0; xc.ncol()];
        descend(&xc, &yc, 1.0, path[0], 1.0, &mut beta, &xx, 1e-7, 100);
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
