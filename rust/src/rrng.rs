//! R's random number stream, reproduced exactly.
//!
//! MORE's MLR path is not deterministic in the way PLS1 is: it draws from R's
//! RNG, so "the same job" gives different answers under different seeds. That
//! is a property of the reference implementation, not a defect, and a port that
//! wants to *agree* with R rather than merely resemble it has to consume the
//! same stream in the same order.
//!
//! The surface is small and was established by tracing a live `more()` run
//! rather than by reading the sources (see `equivalence/rng/trace_rng.R`, which
//! shims `base::sample` and logs every draw). On the MLR path there are exactly
//! four call sites:
//!
//! | site | R | when |
//! | --- | --- | --- |
//! | pair representative | `MORE_MLR.R:811` | `nrow(mycor) == 1`, once |
//! | clique representative | `MORE_MLR.R:860` | once per *complete* component |
//! | star-peel tie-break | `MORE_MLR.R:922` | only on a degree+sum tie |
//! | `cv.glmnet` folds | `foldid = sample(rep(seq(nfolds), length = N))` | 11 per target |
//!
//! `CollinearityFilter2`'s three draws are unreachable (`GetMLR` passes
//! `col.filter = "cor"`), and `p.coef.pls2`'s is the `varSel = "Perm"` path
//! MORE never takes from `runMORE.R`. PLS1 reaches no draw at all, which is why
//! it was already byte-exact.
//!
//! Everything below is a transcription of R 4.6.0's C, not a reimplementation:
//! `do_setseed`'s 50-round scramble and `RNG_Init`'s 625-word fill (`RNG.c`),
//! `MT_genrand` and `fixup` (`RNG.c`), `R_unif_index`/`rbits` under
//! `sample.kind = "Rejection"` (`RNG.c`), and `do_sample`'s without-replacement
//! loop (`random.c`). Verified against R itself in the tests below.

const N: usize = 624;
const M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UPPER_MASK: u32 = 0x8000_0000;
const LOWER_MASK: u32 = 0x7fff_ffff;
const TEMPERING_MASK_B: u32 = 0x9d2c_5680;
const TEMPERING_MASK_C: u32 = 0xefc6_0000;
/// `i2_32m1` in R's RNG.c — 1/(2^32 - 1), used only by `fixup`.
const I2_32M1: f64 = 2.328_306_437_080_797e-10;

/// R's Mersenne-Twister, seeded as `set.seed()` seeds it.
pub struct RRng {
    mt: [u32; N],
    /// `i_seed[0]` in R. Starts at 624 so the first draw regenerates the block.
    mti: usize,
}

impl RRng {
    /// `set.seed(seed)`: `do_setseed`'s initial scrambling, then `RNG_Init`.
    ///
    /// R scrambles 50 times *before* filling, then draws each of the 625 seed
    /// words from the same LCG. `i_seed[0]` is the MT index and is immediately
    /// overwritten by `FixupSeeds(initial = 1)`, so the first word of the fill
    /// is consumed and discarded — dropping it instead shifts the whole state
    /// by one word and every draw is wrong.
    pub fn new(seed: u32) -> Self {
        let mut s = seed;
        for _ in 0..50 {
            s = s.wrapping_mul(69069).wrapping_add(1);
        }
        let mut mt = [0u32; N];
        // j = 0 is i_seed[0] (the MT index): drawn, then discarded by FixupSeeds.
        s = s.wrapping_mul(69069).wrapping_add(1);
        for slot in mt.iter_mut() {
            s = s.wrapping_mul(69069).wrapping_add(1);
            *slot = s;
        }
        Self { mt, mti: N }
    }

    fn genrand(&mut self) -> f64 {
        if self.mti >= N {
            let mag01 = [0u32, MATRIX_A];
            for kk in 0..(N - M) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            for kk in (N - M)..(N - 1) {
                let y = (self.mt[kk] & UPPER_MASK) | (self.mt[kk + 1] & LOWER_MASK);
                self.mt[kk] = self.mt[kk + M - N] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            }
            let y = (self.mt[N - 1] & UPPER_MASK) | (self.mt[0] & LOWER_MASK);
            self.mt[N - 1] = self.mt[M - 1] ^ (y >> 1) ^ mag01[(y & 1) as usize];
            self.mti = 0;
        }
        let mut y = self.mt[self.mti];
        self.mti += 1;
        y ^= y >> 11;
        y ^= (y << 7) & TEMPERING_MASK_B;
        y ^= (y << 15) & TEMPERING_MASK_C;
        y ^= y >> 18;
        f64::from(y) * 2.328_306_436_538_696_3e-10
    }

    /// `unif_rand()` — `MT_genrand` behind `fixup`, which keeps the result
    /// strictly inside (0, 1).
    pub fn unif_rand(&mut self) -> f64 {
        let x = self.genrand();
        if x <= 0.0 {
            return 0.5 * I2_32M1;
        }
        if 1.0 - x <= 0.0 {
            return 1.0 - 0.5 * I2_32M1;
        }
        x
    }

    /// `rbits(bits)`: 16 bits at a time, then masked down.
    ///
    /// The loop condition is `n <= bits`, so `bits == 0` still consumes one
    /// draw and returns 0. That matters: the final step of a full permutation
    /// calls `R_unif_index(1)` and advances the stream even though its answer
    /// is a foregone conclusion.
    fn rbits(&mut self, bits: u32) -> f64 {
        let mut v: u64 = 0;
        let mut n = 0u32;
        while n <= bits {
            let v1 = (self.unif_rand() * 65536.0).floor() as u64;
            v = 65536u64.wrapping_mul(v).wrapping_add(v1);
            n += 16;
        }
        (v & ((1u64 << bits) - 1)) as f64
    }

    /// `R_unif_index(dn)` under `sample.kind = "Rejection"` (R >= 3.6.0, and
    /// the default this deployment runs — `RNGkind()` reports
    /// `Mersenne-Twister / Inversion / Rejection`).
    pub fn unif_index(&mut self, dn: f64) -> f64 {
        if dn <= 0.0 {
            return 0.0;
        }
        let bits = dn.log2().ceil() as u32;
        loop {
            let dv = self.rbits(bits);
            if dn > dv {
                return dv;
            }
        }
    }

    /// `do_sample(n, k, replace = FALSE)` — returns 1-based indices, as R does.
    ///
    /// R takes the `k < 2` branch through a different expression
    /// (`R_unif_index(dn) + 1`) that happens to agree with the general loop for
    /// a single draw; both are reproduced so the code reads like the original.
    pub fn sample_int(&mut self, n: usize, k: usize) -> Vec<usize> {
        if k < 2 {
            return (0..k).map(|_| self.unif_index(n as f64) as usize + 1).collect();
        }
        let mut x: Vec<usize> = (0..n).collect();
        let mut nn = n;
        let mut out = Vec::with_capacity(k);
        for _ in 0..k {
            let j = self.unif_index(nn as f64) as usize;
            out.push(x[j] + 1);
            nn -= 1;
            x[j] = x[nn];
        }
        out
    }

    /// `sample(v, 1)` over a vector of length `n`: the chosen 0-based position.
    pub fn sample_one(&mut self, n: usize) -> usize {
        self.sample_int(n, 1)[0] - 1
    }

    /// `sample(rep(seq(nfolds), length = n))` — `cv.glmnet`'s fold assignment.
    ///
    /// Returns the fold label (1-based) for each observation. Under MORE's own
    /// rule this is leave-one-out below 50 observations, where the labelling
    /// changes nothing — but it consumes the stream either way, and every draw
    /// after it depends on that.
    pub fn fold_ids(&mut self, n: usize, nfolds: usize) -> Vec<usize> {
        let pool: Vec<usize> = (0..n).map(|i| i % nfolds + 1).collect();
        self.sample_int(n, n).into_iter().map(|i| pool[i - 1]).collect()
    }
}

/// The run's single stream, plus the trace that proves it matches R's.
///
/// `MORE_RS_RNG_TRACE=<path>` writes one row per draw in the same shape as
/// `equivalence/rng/trace_rng.R` emits for R, so the two can be diffed line for
/// line. That diff is the acceptance gate for this path: comparing outputs
/// alone cannot distinguish "same answer" from "same answer by luck", and the
/// stream desyncs silently the moment a draw is added or skipped.
pub struct RngStream {
    rng: RRng,
    trace: Option<std::fs::File>,
    seq: usize,
}

impl RngStream {
    pub fn new(seed: u32) -> Self {
        let trace = std::env::var_os("MORE_RS_RNG_TRACE").and_then(|p| {
            use std::io::Write;
            let mut f = std::fs::File::create(&p).ok()?;
            let _ = writeln!(f, "seq\tn\tsize\tinput\tresult\tsite");
            Some(f)
        });
        Self { rng: RRng::new(seed), trace, seq: 0 }
    }

    fn record(&mut self, n: usize, size: usize, input: &str, result: &str, site: &str) {
        self.seq += 1;
        if let Some(f) = self.trace.as_mut() {
            use std::io::Write;
            let _ = writeln!(f, "{}\t{}\t{}\t{}\t{}\t{}", self.seq, n, size, input, result, site);
        }
    }

    /// `sample(names, 1)` — returns the chosen position in `names`.
    pub fn sample_one_of(&mut self, names: &[String], site: &str) -> usize {
        let pick = self.rng.sample_one(names.len());
        let input = names.join(",");
        let result = names[pick].clone();
        self.record(names.len(), 1, &input, &result, site);
        pick
    }

    /// `sample(rep(seq(nfolds), length = n))` — one `cv.glmnet` fold draw.
    pub fn fold_ids(&mut self, n: usize, nfolds: usize, site: &str) -> Vec<usize> {
        let ids = self.rng.fold_ids(n, nfolds);
        let input = (1..=n).map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        let result = ids.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(",");
        self.record(n, n, &input, &result, site);
        ids
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Every expectation below was produced by R 4.6.0 itself, not by hand:
    //   set.seed(123); sample(5, 1); sample(rep(seq(20), length = 20)); ...
    // They are the regression guard that keeps this module honest without
    // requiring R at test time.

    #[test]
    fn unif_rand_matches_r_to_the_last_digit() {
        // set.seed(123); sprintf("%.17g", runif(3))
        let mut r = RRng::new(123);
        let got: Vec<f64> = (0..3).map(|_| r.unif_rand()).collect();
        let want: [f64; 3] = [0.28757752012461424, 0.78830513544380665, 0.40897692181169987];
        for (g, w) in got.iter().zip(want.iter()) {
            assert_eq!(g.to_bits(), w.to_bits(), "got {g:.17e} want {w:.17e}");
        }
    }

    #[test]
    fn single_draws_match_r() {
        // set.seed(123); sample(5,1); <perm>; sample(3,1)
        let mut r = RRng::new(123);
        assert_eq!(r.sample_int(5, 1), vec![3]);
        let _ = r.sample_int(20, 20);
        assert_eq!(r.sample_int(3, 1), vec![2]);

        // set.seed(456); sample(7,1); sample(2,1); sample(100,1)
        let mut r = RRng::new(456);
        assert_eq!(r.sample_int(7, 1), vec![5]);
        assert_eq!(r.sample_int(2, 1), vec![1]);
        assert_eq!(r.sample_int(100, 1), vec![35]);
    }

    #[test]
    fn permutation_matches_r() {
        // set.seed(123); sample(5,1); sample(rep(seq(20), length=20))
        let mut r = RRng::new(123);
        let _ = r.sample_int(5, 1);
        assert_eq!(
            r.sample_int(20, 20),
            vec![14, 3, 10, 11, 5, 4, 20, 6, 9, 18, 16, 19, 12, 1, 15, 7, 17, 13, 8, 2]
        );
    }

    #[test]
    fn fold_ids_match_r_when_folds_are_fewer_than_observations() {
        // set.seed(123); sample(5,1); <perm 20>; sample(3,1);
        // sample(rep(seq(10), length = 36))
        let mut r = RRng::new(123);
        let _ = r.sample_int(5, 1);
        let _ = r.sample_int(20, 20);
        let _ = r.sample_int(3, 1);
        assert_eq!(
            r.fold_ids(36, 10),
            vec![
                7, 9, 9, 4, 7, 1, 6, 1, 2, 5, 10, 3, 10, 5, 3, 6, 4, 6, 2, 5, 8, 8, 5, 8, 1, 2,
                1, 4, 2, 10, 7, 3, 4, 3, 6, 9
            ]
        );
    }

    #[test]
    fn a_one_element_draw_still_advances_the_stream() {
        // R_unif_index(1) takes bits = 0, and rbits' loop runs once at n = 0,
        // so the draw is consumed even though the answer can only be 0. Two
        // streams that disagree about this desync permanently.
        let mut with = RRng::new(123);
        let _ = with.sample_int(1, 1);
        let after_draw = with.unif_rand();

        let mut without = RRng::new(123);
        let first = without.unif_rand();

        assert_ne!(after_draw.to_bits(), first.to_bits());
    }
}
