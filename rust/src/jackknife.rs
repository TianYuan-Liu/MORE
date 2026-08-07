//! Leave-one-out jackknife p-values for PLS coefficients.
//!
//! Ports `p.valuejack` (`../R/auxFunctions.R:652`):
//!
//! ```text
//! k = predI of the main fit
//! for i in 1..n:
//!     refit opls(X[-i, ], scale(y[-i]), scaleC="none", predI=k, crossvalI=1)
//! SE = sqrt( (n-1)/n * sum_i (b_i - b)^2 )
//! p  = 2 * pt(|b / SE|, df = n-1, lower.tail = FALSE)
//! ```
//!
//! Two details that are easy to lose:
//!
//! * the response is re-centred and re-scaled *within each fold*
//!   (`scale(datospls[-i, 1])`), not once up front;
//! * `crossvalI = 1` disables the Q2 computation in the fold fits. With the
//!   component count pinned, Q2 is never consulted, so R leaves it `NaN` and
//!   this port leaves it unread — the difference is unobservable.
//!
//! R pads a fold whose refit dropped variables with zero coefficients and
//! reorders to the main fit's variable order. This port always returns one
//! coefficient per column in the same order, so the reorder is a no-op; a fold
//! that fails to fit at all contributes zeros, which is exactly what the pad does.

use crate::matrix::{scale_in_place, Mat};
use crate::pls::{self, PlsFit};
use statrs::distribution::{ContinuousCDF, StudentsT};

/// Jackknife p-value per coefficient of `main`, in column order.
pub fn p_values(x: &Mat, y: &[f64], main: &PlsFit) -> Vec<f64> {
    let n = x.nrow();
    let p = x.ncol();
    if n < 3 {
        // df = n - 1 must be positive and the fold fits need at least two rows.
        return vec![f64::NAN; p];
    }

    // Sum of squared deviations of the fold coefficients from the main fit.
    let mut ss = vec![0.0; p];
    for i in 0..n {
        let xi = x.without_rows(&[i]);
        let mut yi: Vec<f64> = (0..n).filter(|&j| j != i).map(|j| y[j]).collect();
        scale_in_place(&mut yi);

        let fold = pls::fit(&xi, &yi, Some(main.n_comp), 1);
        match fold {
            Some(f) => {
                for j in 0..p {
                    let d = f.coefficients[j] - main.coefficients[j];
                    ss[j] += d * d;
                }
            }
            None => {
                // Mirrors R's zero-padding of variables absent from a fold refit.
                for j in 0..p {
                    let d = main.coefficients[j];
                    ss[j] += d * d;
                }
            }
        }
    }

    let df = n as f64 - 1.0;
    let scale = (n as f64 - 1.0) / n as f64;
    let t_dist = match StudentsT::new(0.0, 1.0, df) {
        Ok(d) => d,
        Err(_) => return vec![f64::NAN; p],
    };

    (0..p)
        .map(|j| {
            let se = (scale * ss[j]).sqrt();
            if !(se > 0.0) {
                // A coefficient that never moved has no jackknife spread. R
                // divides by zero here and gets Inf, whose two-sided tail is 0
                // for a non-zero coefficient and NaN for 0/0.
                return if main.coefficients[j] == 0.0 { f64::NAN } else { 0.0 };
            }
            let t = (main.coefficients[j] / se).abs();
            2.0 * (1.0 - t_dist.cdf(t))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matrix::Mat;

    /// x1 drives y strongly; x2 barely contributes.
    ///
    /// The response is scaled here because that is how the caller supplies it:
    /// `ResultsPerTargetF.i:107` fits `opls(X, scale(response))`, and
    /// `p_values` re-scales the response within each fold. Handing the main fit
    /// an unscaled response while the folds see a scaled one puts the two
    /// coefficient sets on different scales and inflates every SE.
    fn design() -> (Mat, Vec<f64>) {
        let x1 = vec![-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 0.25, -0.25, 0.75];
        let x2 = vec![0.7, -0.3, 1.1, -1.2, 0.4, -0.9, 0.2, 1.3, -1.1, 0.6];
        let mut y: Vec<f64> = x1
            .iter()
            .zip(&x2)
            .map(|(a, b)| 2.0 * a + 0.01 * b)
            .collect();
        scale_in_place(&mut y);
        (Mat::from_columns(&[x1, x2]), y)
    }

    #[test]
    fn a_strong_predictor_gets_a_small_p_value() {
        let (x, y) = design();
        let main = pls::fit(&x, &y, Some(1), 7).expect("model");
        let p = p_values(&x, &y, &main);
        assert!(p[0] < 0.05, "p for the driver was {}", p[0]);
    }

    #[test]
    fn p_values_lie_in_the_unit_interval() {
        let (x, y) = design();
        let main = pls::fit(&x, &y, Some(1), 7).expect("model");
        for v in p_values(&x, &y, &main) {
            assert!((0.0..=1.0).contains(&v) || v.is_nan(), "out of range: {v}");
        }
    }

    #[test]
    fn one_p_value_is_returned_per_column() {
        let (x, y) = design();
        let main = pls::fit(&x, &y, Some(1), 7).expect("model");
        assert_eq!(p_values(&x, &y, &main).len(), x.ncol());
    }

    #[test]
    fn too_few_observations_yield_nan_rather_than_a_bogus_test() {
        let x = Mat::from_columns(&[vec![1.0, 2.0]]);
        let y = vec![1.0, 2.0];
        let main = PlsFit {
            n_comp: 1,
            coefficients: vec![1.0],
            vip: vec![1.0],
            r2y_cum: 1.0,
            q2_cum: 1.0,
            rmsee: 0.0,
            fitted: vec![1.0, 2.0],
        };
        assert!(p_values(&x, &y, &main)[0].is_nan());
    }
}
