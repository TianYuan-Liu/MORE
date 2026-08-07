//! Dense column-major matrix.
//!
//! Column-major because the PLS inner loop is dominated by `X'u` and `X't`
//! (one dot product per column) and by per-column deflation. Only `t = Xw`
//! reads across rows, and it is the cheapest of the three.
//!
//! No BLAS. Phase-1 native profiling of the R pipeline put BLAS at 0.1% of
//! runtime, and the per-target matrices here are ~20 rows by a few hundred
//! columns, where call overhead exceeds any kernel win. Dropping the
//! dependency also makes the static musl build trivial.

/// Column-major dense matrix of `f64`.
#[derive(Clone, Debug, PartialEq)]
pub struct Mat {
    nrow: usize,
    ncol: usize,
    /// `data[j * nrow + i]` is element (i, j).
    data: Vec<f64>,
}

impl Mat {
    pub fn zeros(nrow: usize, ncol: usize) -> Self {
        Mat { nrow, ncol, data: vec![0.0; nrow * ncol] }
    }

    /// Build from column slices. Every column must have the same length.
    pub fn from_columns(cols: &[Vec<f64>]) -> Self {
        let ncol = cols.len();
        let nrow = cols.first().map_or(0, |c| c.len());
        debug_assert!(cols.iter().all(|c| c.len() == nrow));
        let mut data = Vec::with_capacity(nrow * ncol);
        for col in cols {
            data.extend_from_slice(col);
        }
        Mat { nrow, ncol, data }
    }

    /// Build from a row-major slice — convenient for literals in tests.
    pub fn from_rows(nrow: usize, ncol: usize, rows: &[f64]) -> Self {
        assert_eq!(rows.len(), nrow * ncol);
        let mut m = Mat::zeros(nrow, ncol);
        for i in 0..nrow {
            for j in 0..ncol {
                m.set(i, j, rows[i * ncol + j]);
            }
        }
        m
    }

    #[inline]
    pub fn nrow(&self) -> usize {
        self.nrow
    }

    #[inline]
    pub fn ncol(&self) -> usize {
        self.ncol
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[j * self.nrow + i]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[j * self.nrow + i] = v;
    }

    #[inline]
    pub fn col(&self, j: usize) -> &[f64] {
        &self.data[j * self.nrow..(j + 1) * self.nrow]
    }

    #[inline]
    pub fn col_mut(&mut self, j: usize) -> &mut [f64] {
        let n = self.nrow;
        &mut self.data[j * n..(j + 1) * n]
    }

    /// `X' v` — one dot product per column. Length `ncol`.
    pub fn t_mul_vec(&self, v: &[f64]) -> Vec<f64> {
        debug_assert_eq!(v.len(), self.nrow);
        (0..self.ncol)
            .map(|j| dot(self.col(j), v))
            .collect()
    }

    /// `X v` — length `nrow`. Accumulated column-wise so the column-major
    /// layout is still walked contiguously.
    pub fn mul_vec(&self, v: &[f64]) -> Vec<f64> {
        debug_assert_eq!(v.len(), self.ncol);
        let mut out = vec![0.0; self.nrow];
        for j in 0..self.ncol {
            let vj = v[j];
            if vj == 0.0 {
                continue;
            }
            for (o, x) in out.iter_mut().zip(self.col(j)) {
                *o += vj * x;
            }
        }
        out
    }

    /// In-place rank-one deflation `X -= t p'`.
    pub fn deflate(&mut self, t: &[f64], p: &[f64]) {
        debug_assert_eq!(t.len(), self.nrow);
        debug_assert_eq!(p.len(), self.ncol);
        for j in 0..self.ncol {
            let pj = p[j];
            if pj == 0.0 {
                continue;
            }
            for (x, tv) in self.col_mut(j).iter_mut().zip(t) {
                *x -= tv * pj;
            }
        }
    }

    /// Sum of squares over every element, skipping NaN — matches R's
    /// `sum(x^2, na.rm = TRUE)`.
    pub fn sum_squares(&self) -> f64 {
        self.data.iter().filter(|v| !v.is_nan()).map(|v| v * v).sum()
    }

    /// A copy with `rows` removed — the jackknife's `X[-i, ]`.
    pub fn without_rows(&self, drop: &[usize]) -> Mat {
        let keep: Vec<usize> = (0..self.nrow).filter(|i| !drop.contains(i)).collect();
        self.select_rows(&keep)
    }

    /// A copy holding only `keep`, in the order given.
    pub fn select_rows(&self, keep: &[usize]) -> Mat {
        let mut out = Mat::zeros(keep.len(), self.ncol);
        for j in 0..self.ncol {
            let src = self.col(j);
            let dst = out.col_mut(j);
            for (d, &i) in dst.iter_mut().zip(keep) {
                *d = src[i];
            }
        }
        out
    }

    /// A copy holding only the named columns, in the order given.
    pub fn select_cols(&self, keep: &[usize]) -> Mat {
        let mut out = Mat::zeros(self.nrow, keep.len());
        for (jd, &js) in keep.iter().enumerate() {
            out.col_mut(jd).copy_from_slice(self.col(js));
        }
        out
    }

    /// Per-column standard deviation, `na.rm = FALSE` — an NA anywhere in the
    /// column yields NaN, which callers must treat as "keep" (see SPEC §1 step 6:
    /// R's `is.na(sd) | sd > 0` retains such columns).
    pub fn col_sd(&self) -> Vec<f64> {
        (0..self.ncol).map(|j| sd(self.col(j))).collect()
    }
}

/// Dot product, NaN-propagating (callers rely on NaN reaching them).
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[inline]
pub fn norm2(a: &[f64]) -> f64 {
    dot(a, a)
}

/// Sample standard deviation with denominator `n - 1`, matching R's `sd()`.
/// Returns NaN if any element is NaN, and 0.0 for a single observation.
pub fn sd(x: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return f64::NAN;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let ss: f64 = x.iter().map(|v| (v - mean) * (v - mean)).sum();
    (ss / (n as f64 - 1.0)).sqrt()
}

/// R's `signif(x, digits)`.
///
/// ropls rounds `R2X`, `R2Y`, `Q2`, `RMSEE` and their cumulative forms to three
/// significant digits on the way out (`ropls:919-924`), *after* the
/// component-selection loop has already tested the full-precision values. MORE
/// reads only the rounded numbers, and one of them becomes the `R2` column of
/// `MORE_rpc_*.tab`, so the port has to round identically or the output file
/// differs textually.
///
/// R's C implementation rounds through `nearbyint`, i.e. ties-to-even, which is
/// `round_ties_even` here rather than `round`. Exact ties are unreachable for
/// values that came out of a floating-point fit, so the distinction is
/// defensive rather than load-bearing.
pub fn signif(x: f64, digits: i32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let ax = x.abs();
    let e10 = digits - 1 - ax.log10().floor() as i32;
    let pow10 = 10f64.powi(e10);
    x.signum() * (ax * pow10).round_ties_even() / pow10
}

/// Centre and scale to unit variance, matching R's `scale(x, TRUE, TRUE)`.
/// A constant column becomes all-NaN in R (division by sd = 0); reproduced here.
pub fn scale_in_place(x: &mut [f64]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let mean = x.iter().sum::<f64>() / n as f64;
    let s = sd(x);
    for v in x.iter_mut() {
        *v = (*v - mean) / s;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-12, "{a} != {b}");
    }

    #[test]
    fn from_rows_and_get_agree() {
        // 2x3 matrix laid out row-major in the literal.
        let m = Mat::from_rows(2, 3, &[1., 2., 3., 4., 5., 6.]);
        approx(m.get(0, 0), 1.);
        approx(m.get(0, 2), 3.);
        approx(m.get(1, 0), 4.);
        approx(m.get(1, 2), 6.);
        assert_eq!(m.col(1), &[2., 5.]);
    }

    #[test]
    fn transpose_mul_matches_hand_computation() {
        let m = Mat::from_rows(2, 3, &[1., 2., 3., 4., 5., 6.]);
        // X' v with v = (1, 2): columns dotted with v.
        let got = m.t_mul_vec(&[1., 2.]);
        assert_eq!(got, vec![1. + 8., 2. + 10., 3. + 12.]);
    }

    #[test]
    fn mul_vec_matches_hand_computation() {
        let m = Mat::from_rows(2, 3, &[1., 2., 3., 4., 5., 6.]);
        let got = m.mul_vec(&[1., 0., 2.]);
        assert_eq!(got, vec![1. + 0. + 6., 4. + 0. + 12.]);
    }

    #[test]
    fn deflation_removes_the_rank_one_term() {
        let mut m = Mat::from_rows(2, 2, &[1., 2., 3., 4.]);
        m.deflate(&[1., 1.], &[1., 2.]);
        approx(m.get(0, 0), 0.);
        approx(m.get(0, 1), 0.);
        approx(m.get(1, 0), 2.);
        approx(m.get(1, 1), 2.);
    }

    #[test]
    fn sd_uses_n_minus_one_like_r() {
        // R: sd(c(1,2,3,4)) == 1.290994
        approx(sd(&[1., 2., 3., 4.]), 1.2909944487358056);
    }

    #[test]
    fn sd_of_a_constant_column_is_zero() {
        approx(sd(&[2., 2., 2.]), 0.0);
    }

    #[test]
    fn sd_propagates_nan() {
        assert!(sd(&[1., f64::NAN, 3.]).is_nan());
    }

    #[test]
    fn scaling_gives_zero_mean_unit_sd() {
        let mut x = vec![1., 2., 3., 4.];
        scale_in_place(&mut x);
        approx(x.iter().sum::<f64>(), 0.0);
        approx(sd(&x), 1.0);
    }

    #[test]
    fn scaling_a_constant_column_yields_nan_like_r() {
        let mut x = vec![5., 5., 5.];
        scale_in_place(&mut x);
        assert!(x.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn row_selection_drops_the_named_row() {
        let m = Mat::from_rows(3, 2, &[1., 2., 3., 4., 5., 6.]);
        let got = m.without_rows(&[1]);
        assert_eq!(got.nrow(), 2);
        assert_eq!(got.col(0), &[1., 5.]);
        assert_eq!(got.col(1), &[2., 6.]);
    }

    #[test]
    fn signif_matches_r_reference_values() {
        // Left-hand sides taken from R: signif(x, 3).
        approx(signif(0.9739954, 3), 0.974);
        approx(signif(0.8275, 3), 0.828);
        approx(signif(0.17512, 3), 0.175);
        approx(signif(0.0001234567, 3), 0.000123);
        approx(signif(123456.0, 3), 123000.0);
        approx(signif(-0.0005555, 3), -0.000556);
        approx(signif(0.1235, 3), 0.124);
        approx(signif(0.1245, 3), 0.124);
        approx(signif(1.0, 3), 1.0);
        approx(signif(0.0, 3), 0.0);
    }

    #[test]
    fn signif_leaves_a_value_with_fewer_digits_alone() {
        approx(signif(2.5e-8, 3), 2.5e-8);
    }

    #[test]
    fn signif_passes_non_finite_values_through() {
        assert!(signif(f64::NAN, 3).is_nan());
        assert_eq!(signif(f64::INFINITY, 3), f64::INFINITY);
    }

    #[test]
    fn sum_squares_skips_nan_like_r_na_rm() {
        let m = Mat::from_rows(1, 3, &[2., f64::NAN, 3.]);
        approx(m.sum_squares(), 13.0);
    }
}
