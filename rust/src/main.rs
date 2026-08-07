//! `more-rs` — Rust port of the MORE regulatory-model kernel.
//!
//! Drop-in for `PaintomicsServer/src/common/bioscripts/runMORE.R`: same CLI
//! options, same output filenames, same file formats. See `SPEC.md` for the
//! rule-by-rule mapping onto the R reference in `../R/`, which is never edited
//! and stays the oracle for the equivalence harness.

mod cli;
mod jackknife;
mod matrix;
mod pls;

#[cfg(test)]
mod oracle_test;

use std::process::ExitCode;

fn main() -> ExitCode {
    let opts = match cli::Options::parse_args() {
        Ok(o) => o,
        Err(e) => {
            eprintln!("MORE ERROR: {e}");
            return ExitCode::FAILURE;
        }
    };

    // The pipeline stages land in subsequent commits; the CLI contract is
    // pinned first so the equivalence harness can drive the binary from the
    // start and fail loudly on any option drift.
    eprintln!(
        "MORE: method={} alpha={} vip={} omics={:?}",
        opts.method, opts.alpha, opts.vip, opts.omic_names
    );
    eprintln!("MORE ERROR: pipeline not implemented yet");
    ExitCode::FAILURE
}
