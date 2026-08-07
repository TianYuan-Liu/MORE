//! PLS1 by NIPALS, reproducing `ropls::opls` as MORE calls it.
//!
//! MORE always calls `ropls::opls(X, y, scaleC = "none", crossvalI = cross,
//! permI = 0)` with `predI` left `NA` (autofit), falling back to `predI = 1`
//! when autofit yields no significant component. `scaleC = "none"` means ropls
//! neither centres nor scales — MORE has already done both — so `xMeanVn = 0`,
//! `xSdVn = 1`, `yMeanVn = 0`, `ySdVn = 1` and the predicted response needs no
//! unscaling (`ropls:97-108, 212-225, 464`).
//!
//! Everything here is deterministic. ropls' cross-validation folds are
//! `split(1:n, rep(1:crossvalI, length = n))` — interleaved, not sampled — and
//! with a single response the NIPALS inner loop exits on its first pass
//! (`ropls:281`). No RNG is reachable on this path, so a divergence from R here
//! is a bug, never seed noise.
//!
//! See `../SPEC.md` §3 for the rule-by-rule mapping to the R source.

use crate::matrix::{dot, norm2, signif, Mat};

/// Threshold on `R2Y` below which a component is rejected (`ropls:394`).
const R2Y_MIN: f64 = 0.01;
/// Hard cap on components in autofit mode (`ropls:56`).
const AUT_MAX_COMPONENTS: usize = 10;

/// A fitted PLS1 model, carrying only what MORE reads back.
#[derive(Clone, Debug)]
pub struct PlsFit {
    /// Number of retained components — ropls' `summaryDF$pre`.
    pub n_comp: usize,
    /// Regression coefficients on the supplied (already scaled) X, one per column.
    pub coefficients: Vec<f64>,
    /// Variable importance in projection, one per column.
    pub vip: Vec<f64>,
    /// `R2Y(cum)` at the retained component count.
    pub r2y_cum: f64,
    /// `Q2(cum)` at the retained component count.
    pub q2_cum: f64,
    /// `RMSEE` — ropls' error-of-estimate, adjusted for model dimension.
    pub rmsee: f64,
    /// Fitted response, same scale as the `y` passed in.
    pub fitted: Vec<f64>,
}

impl PlsFit {
    pub fn residuals(&self, y: &[f64]) -> Vec<f64> {
        y.iter().zip(&self.fitted).map(|(a, b)| a - b).collect()
    }
}

/// Fit PLS1.
///
/// `pred_i` is `None` for ropls' autofit, or `Some(k)` to force exactly `k`
/// components (MORE's retry path, and the jackknife, which pins the component
/// count to the main fit).
///
/// Returns `None` when no model exists. That covers both of R's ways of
/// failing: autofit retaining zero components (`ropls:418`, which yields an
/// object with an empty `modelDF` that MORE tests for), and a degenerate fit
/// where a norm underflows to zero — in R the resulting `NaN` reaches
/// `if (R2Y < 0.01)` and raises "missing value where TRUE/FALSE needed", which
/// MORE's `try()` swallows into the same "no model" outcome.
pub fn fit(x: &Mat, y: &[f64], pred_i: Option<usize>, crossval: usize) -> Option<PlsFit> {
    let n = x.nrow();
    let p = x.ncol();
    if n == 0 || p == 0 || y.len() != n {
        return None;
    }

    let autofit = pred_i.is_none();
    let aut_max = AUT_MAX_COMPONENTS.min(n).min(p);
    let max_comp = match pred_i {
        Some(k) => k.min(p).max(1),
        None => aut_max,
    };
    if max_comp == 0 {
        return None;
    }

    // `ru1ThrN` (ropls:237). For PLS (no orthogonal components) the Q2 floor is
    // 0.05 on small designs and 0 once there are more than 100 observations.
    let q2_floor = if n > 100 { 0.0 } else { 0.05 };

    let ssx_tot = x.sum_squares();
    let ssy_tot: f64 = y.iter().filter(|v| !v.is_nan()).map(|v| v * v).sum();
    if !(ssy_tot > 0.0) {
        return None;
    }

    let folds = cv_folds(n, crossval);

    let mut xn = x.clone();
    let mut yn = y.to_vec();
    let mut rss = ssy_tot;

    let mut w_mat: Vec<Vec<f64>> = Vec::with_capacity(max_comp); // p per component
    let mut t_mat: Vec<Vec<f64>> = Vec::with_capacity(max_comp); // n per component
    let mut p_mat: Vec<Vec<f64>> = Vec::with_capacity(max_comp); // p per component
    let mut c_vec: Vec<f64> = Vec::with_capacity(max_comp); // scalar per component
    let mut r2y: Vec<f64> = Vec::with_capacity(max_comp);
    let mut q2: Vec<f64> = Vec::with_capacity(max_comp);

    for _h in 0..max_comp {
        // NIPALS, single response: u = y, and the loop breaks after one pass.
        let uu = norm2(&yn);
        if !(uu > 0.0) {
            return None;
        }
        let mut w = xn.t_mul_vec(&yn);
        for v in w.iter_mut() {
            *v /= uu;
        }
        let wnorm = norm2(&w).sqrt();
        if !(wnorm > 0.0) {
            return None;
        }
        for v in w.iter_mut() {
            *v /= wnorm;
        }

        let t = xn.mul_vec(&w);
        let tt = norm2(&t);
        if !(tt > 0.0) {
            return None;
        }
        let c = dot(&yn, &t) / tt;
        let mut pv = xn.t_mul_vec(&t);
        for v in pv.iter_mut() {
            *v /= tt;
        }

        // sum((t p')^2) == (t't)(p'p); sum((t c)^2) == (t't) c^2.
        let _r2x_h = tt * norm2(&pv) / ssx_tot;
        let r2y_h = tt * c * c / ssy_tot;

        // Cross-validated PRESS for this component (ropls:322-392).
        let mut press = 0.0;
        for fold in &folds {
            press += fold_press(&xn, &yn, fold);
        }
        let q2_h = 1.0 - press / rss;

        // Significance (ropls:394-403). R2Y is tested first; a component failing
        // either test is discarded, and in autofit mode ends the search.
        let significant = r2y_h >= R2Y_MIN && q2_h >= q2_floor;
        if autofit && !significant {
            break;
        }

        // Deflate for the next component.
        let mut resid = 0.0;
        for (yv, tv) in yn.iter_mut().zip(&t) {
            *yv -= tv * c;
            resid += *yv * *yv;
        }
        rss = resid;
        xn.deflate(&t, &pv);

        w_mat.push(w);
        t_mat.push(t);
        p_mat.push(pv);
        c_vec.push(c);
        r2y.push(r2y_h);
        q2.push(q2_h);
    }

    let n_comp = w_mat.len();
    if n_comp == 0 {
        return None;
    }

    // R = W (P'W)^-1, with R = W for a single component (ropls:450-457).
    let r_mat = if n_comp == 1 {
        w_mat.clone()
    } else {
        let mut pw = vec![0.0; n_comp * n_comp]; // row-major, n_comp <= 10
        for a in 0..n_comp {
            for b in 0..n_comp {
                pw[a * n_comp + b] = dot(&p_mat[a], &w_mat[b]);
            }
        }
        let inv = invert(&pw, n_comp)?;
        // R[, h] = sum_k W[, k] * inv[k, h]
        (0..n_comp)
            .map(|h| {
                let mut col = vec![0.0; p];
                for k in 0..n_comp {
                    let f = inv[k * n_comp + h];
                    if f == 0.0 {
                        continue;
                    }
                    for (o, wv) in col.iter_mut().zip(&w_mat[k]) {
                        *o += f * wv;
                    }
                }
                col
            })
            .collect()
    };

    // B = R C'.
    let mut coefficients = vec![0.0; p];
    for h in 0..n_comp {
        let c = c_vec[h];
        for (b, r) in coefficients.iter_mut().zip(&r_mat[h]) {
            *b += r * c;
        }
    }

    // Fitted response = T C'; no unscaling needed under scaleC = "none".
    let mut fitted = vec![0.0; n];
    for h in 0..n_comp {
        let c = c_vec[h];
        for (f, t) in fitted.iter_mut().zip(&t_mat[h]) {
            *f += t * c;
        }
    }

    // VIP (ropls:883-886): ssy[h] = ||t_h c_h||^2 = (t_h't_h) c_h^2.
    let ssy: Vec<f64> = (0..n_comp)
        .map(|h| norm2(&t_mat[h]) * c_vec[h] * c_vec[h])
        .collect();
    let ssy_sum: f64 = ssy.iter().sum();
    let vip: Vec<f64> = (0..p)
        .map(|j| {
            let acc: f64 = (0..n_comp).map(|h| w_mat[h][j] * w_mat[h][j] * ssy[h]).sum();
            (p as f64 * acc / ssy_sum).sqrt()
        })
        .collect();

    // Accumulated at full precision — the significance tests above already ran
    // on the unrounded per-component values, exactly as in R.
    let r2y_cum: f64 = r2y.iter().sum();
    let q2_cum: f64 = 1.0 - q2.iter().map(|v| 1.0 - v).product::<f64>();

    // RMSEE (ropls:476): sqrt(mean((y - yhat)^2) * n / (n - (1 + n_comp))).
    let mse: f64 = y
        .iter()
        .zip(&fitted)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        / n as f64;
    let denom = n as f64 - (1.0 + n_comp as f64);
    let rmsee = (mse * n as f64 / denom).sqrt();

    // ropls truncates these three to three significant digits before any caller
    // sees them (ropls:919-924), and MORE copies R2Y(cum) straight into the R2
    // column of MORE_rpc_*.tab. Round here so the port's output file matches
    // byte for byte. Coefficients, VIP and fitted values are NOT rounded.
    Some(PlsFit {
        n_comp,
        coefficients,
        vip,
        r2y_cum: signif(r2y_cum, 3),
        q2_cum: signif(q2_cum, 3),
        rmsee: signif(rmsee, 3),
        fitted,
    })
}

/// ropls' cross-validation folds: `split(1:n, rep(1:crossvalI, length = n))`.
/// Observation `i` (0-based) lands in fold `i % crossval`. Deterministic —
/// there is no sampling anywhere in this path.
fn cv_folds(n: usize, crossval: usize) -> Vec<Vec<usize>> {
    let k = crossval.clamp(1, n.max(1));
    let mut folds = vec![Vec::new(); k];
    for i in 0..n {
        folds[i % k].push(i);
    }
    folds.retain(|f| !f.is_empty());
    folds
}

/// PRESS contribution of one held-out fold: fit a single component on the
/// remaining rows, then predict the held-out rows as `X_out w c`.
fn fold_press(xn: &Mat, yn: &[f64], out: &[usize]) -> f64 {
    let keep: Vec<usize> = (0..xn.nrow()).filter(|i| !out.contains(i)).collect();
    if keep.is_empty() {
        return 0.0;
    }
    let ck_x = xn.select_rows(&keep);
    let ck_y: Vec<f64> = keep.iter().map(|&i| yn[i]).collect();

    let uu = norm2(&ck_y);
    if !(uu > 0.0) {
        return 0.0;
    }
    let mut ck_w = ck_x.t_mul_vec(&ck_y);
    for v in ck_w.iter_mut() {
        *v /= uu;
    }
    let wnorm = norm2(&ck_w).sqrt();
    if !(wnorm > 0.0) {
        return 0.0;
    }
    for v in ck_w.iter_mut() {
        *v /= wnorm;
    }
    let ck_t = ck_x.mul_vec(&ck_w);
    let tt = norm2(&ck_t);
    if !(tt > 0.0) {
        return 0.0;
    }
    let ck_c = dot(&ck_y, &ck_t) / tt;

    out.iter()
        .map(|&i| {
            let score: f64 = (0..xn.ncol()).map(|j| xn.get(i, j) * ck_w[j]).sum();
            let err = yn[i] - score * ck_c;
            err * err
        })
        .sum()
}

/// Invert a small dense row-major matrix by Gauss-Jordan with partial pivoting.
/// Only ever called on `P'W`, which is at most 10x10.
fn invert(a: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut m = a.to_vec();
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        inv[i * n + i] = 1.0;
    }
    for col in 0..n {
        let (pivot, _) = (col..n).fold((col, 0.0), |(bi, bv), r| {
            let v = m[r * n + col].abs();
            if v > bv {
                (r, v)
            } else {
                (bi, bv)
            }
        });
        if m[pivot * n + col].abs() < 1e-300 {
            return None;
        }
        if pivot != col {
            for j in 0..n {
                m.swap(pivot * n + j, col * n + j);
                inv.swap(pivot * n + j, col * n + j);
            }
        }
        let d = m[col * n + col];
        for j in 0..n {
            m[col * n + j] /= d;
            inv[col * n + j] /= d;
        }
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r * n + col];
            if f == 0.0 {
                continue;
            }
            for j in 0..n {
                m[r * n + j] -= f * m[col * n + j];
                inv[r * n + j] -= f * inv[col * n + j];
            }
        }
    }
    Some(inv)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "{a} != {b} (tol {tol})");
    }

    /// A design where y is an exact linear function of the first column.
    ///
    /// The columns are correlated (`x1 . x2 = -1.7`), so a *one*-component fit
    /// is deliberately shrunk — PLS1 only coincides with OLS once it has as
    /// many components as columns. Tests that want the exact relation back
    /// therefore ask for the full rank.
    fn exact_design() -> (Mat, Vec<f64>) {
        let x1 = vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 0.0];
        let x2 = vec![0.7, -0.3, 1.1, -1.2, 0.4, -0.9, 0.2, 0.0];
        let y: Vec<f64> = x1.iter().map(|v| 2.0 * v).collect();
        (Mat::from_columns(&[x1, x2]), y)
    }

    /// An orthogonal design plus a response that lies almost entirely outside
    /// the column space, so the first component's R2Y falls under ropls' 1%
    /// floor. `z` is the Hadamard product of the two columns, hence orthogonal
    /// to both.
    fn uninformative_design() -> (Mat, Vec<f64>) {
        let x1 = vec![1., -1., 1., -1., 1., -1., 1., -1.];
        let x2 = vec![1., 1., -1., -1., 1., 1., -1., -1.];
        let z = [1., -1., -1., 1., 1., -1., -1., 1.];
        let y: Vec<f64> = x1.iter().zip(&z).map(|(a, b)| 0.05 * a + b).collect();
        (Mat::from_columns(&[x1, x2]), y)
    }

    #[test]
    fn folds_are_interleaved_not_sampled() {
        // rep(1:3, length = 7) -> 1 2 3 1 2 3 1
        let folds = cv_folds(7, 3);
        assert_eq!(folds[0], vec![0, 3, 6]);
        assert_eq!(folds[1], vec![1, 4]);
        assert_eq!(folds[2], vec![2, 5]);
    }

    #[test]
    fn folds_never_exceed_the_observation_count() {
        let folds = cv_folds(3, 7);
        assert_eq!(folds.len(), 3);
        assert!(folds.iter().all(|f| f.len() == 1));
    }

    #[test]
    fn an_exact_linear_relation_is_recovered_at_full_rank() {
        let (x, y) = exact_design();
        let fit = fit(&x, &y, Some(2), 4).expect("model");
        approx(fit.coefficients[0], 2.0, 1e-9);
        approx(fit.coefficients[1], 0.0, 1e-9);
        for (f, t) in fit.fitted.iter().zip(&y) {
            approx(*f, *t, 1e-9);
        }
    }

    #[test]
    fn one_component_shrinks_towards_the_correlated_column() {
        // Not a defect: a rank-deficient PLS fit is biased by construction.
        // Pinned so that any future change to the deflation is visible.
        let (x, y) = exact_design();
        let fit = fit(&x, &y, Some(1), 4).expect("model");
        assert!(fit.coefficients[0] < 2.0, "{:?}", fit.coefficients);
        assert!(fit.coefficients[1] != 0.0);
    }

    #[test]
    fn a_perfect_fit_has_r2y_of_one() {
        let (x, y) = exact_design();
        let fit = fit(&x, &y, Some(2), 4).expect("model");
        approx(fit.r2y_cum, 1.0, 1e-9);
    }

    #[test]
    fn vip_favours_the_informative_column() {
        let (x, y) = exact_design();
        let fit = fit(&x, &y, Some(1), 4).expect("model");
        assert!(fit.vip[0] > fit.vip[1], "{:?}", fit.vip);
    }

    #[test]
    fn vip_squares_average_to_one_across_variables() {
        // The standard VIP identity: mean(VIP^2) == 1.
        let (x, y) = exact_design();
        let fit = fit(&x, &y, Some(1), 4).expect("model");
        let mean_sq: f64 = fit.vip.iter().map(|v| v * v).sum::<f64>() / fit.vip.len() as f64;
        approx(mean_sq, 1.0, 1e-9);
    }

    #[test]
    fn a_response_below_the_r2y_floor_retains_no_component_under_autofit() {
        // R2Y of the first component is 0.05^2 * 8 / 8.02 = 0.0025, under
        // ropls' 1% floor, so autofit keeps nothing. This is the path where
        // MORE falls back to predI = 1.
        let (x, y) = uninformative_design();
        assert!(fit(&x, &y, None, 4).is_none());
    }

    #[test]
    fn the_same_response_still_fits_when_a_component_is_forced() {
        // MORE's retry: the R2Y floor only applies in autofit mode.
        let (x, y) = uninformative_design();
        let fit = fit(&x, &y, Some(1), 4).expect("forced fit");
        assert_eq!(fit.n_comp, 1);
    }

    #[test]
    fn a_constant_response_yields_no_model() {
        let (x, _) = exact_design();
        let y = vec![0.0; 8];
        assert!(fit(&x, &y, None, 4).is_none());
    }

    #[test]
    fn forcing_more_components_never_exceeds_the_column_count() {
        let (x, y) = exact_design();
        let fit = fit(&x, &y, Some(5), 4).expect("model");
        assert!(fit.n_comp <= x.ncol());
    }

    #[test]
    fn inverting_identity_returns_identity() {
        let inv = invert(&[1., 0., 0., 1.], 2).unwrap();
        assert_eq!(inv, vec![1., 0., 0., 1.]);
    }

    #[test]
    fn inverting_a_singular_matrix_fails_rather_than_returning_garbage() {
        assert!(invert(&[1., 2., 2., 4.], 2).is_none());
    }

    #[test]
    fn inversion_round_trips_a_general_matrix() {
        let a = [4., 7., 2., 6.];
        let inv = invert(&a, 2).unwrap();
        // a * inv == I
        approx(a[0] * inv[0] + a[1] * inv[2], 1.0, 1e-12);
        approx(a[0] * inv[1] + a[1] * inv[3], 0.0, 1e-12);
        approx(a[2] * inv[0] + a[3] * inv[2], 0.0, 1e-12);
        approx(a[2] * inv[1] + a[3] * inv[3], 1.0, 1e-12);
    }

    #[test]
    fn residuals_are_the_response_minus_the_fit() {
        let (x, y) = exact_design();
        let fit = fit(&x, &y, Some(2), 4).expect("model");
        for r in fit.residuals(&y) {
            approx(r, 0.0, 1e-9);
        }
    }
}
