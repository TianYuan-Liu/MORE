//! Two inputs that used to exit 0 over an empty result.
//!
//! Both are end-to-end on purpose. Neither defect lives in a function: every
//! unit did what it was asked, and `run` simply never asked whether the job
//! had produced anything before printing "Analysis complete."
//!
//! What the user saw in both cases was a MORE job that finished, a green
//! Step 2, and a Step 3 panel with nothing in it -- no error anywhere, and
//! nothing to distinguish "your data has no regulation in it" from "MORE
//! could not model your data at all". The R engine stops with an error on
//! both inputs, which is how they were found.

use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};

fn write(dir: &PathBuf, name: &str, body: &str) -> String {
    let p = dir.join(name);
    fs::write(&p, body).unwrap();
    p.to_string_lossy().into_owned()
}

/// Every input file comes from the caller: these tests differ from each other
/// only in the data, and a shared fixture would hide which part is under test.
fn run_more(
    dir: &PathBuf,
    design: &str,
    target: &str,
    regs: &str,
    assoc: &str,
    method: &str,
) -> Output {
    let out = dir.join("out");
    let target = write(dir, "target.tab", target);
    let regs = write(dir, "regs.tab", regs);
    let assoc = write(dir, "assoc.tab", assoc);
    let design = write(dir, "design.tab", design);
    Command::new(env!("CARGO_BIN_EXE_more-rs"))
        .args([
            "--target_file", &target,
            "--condition_file", &design,
            "--omic_names", "TF",
            "--data_files", &regs,
            "--assoc_files", &assoc,
            "--min_variation", "NA",
            "--method", method,
            "--output_dir", &out.to_string_lossy(),
            "--date_seed", "seed",
        ])
        .output()
        .expect("more-rs did not start")
}

const TWO_TARGETS: &str = "GeneID\tS1\tS2\tS3\tS4\tS5\tS6\n\
                           G1\t1.0\t1.2\t0.9\t5.0\t5.2\t4.8\n\
                           G2\t2.0\t2.1\t1.9\t8.0\t8.1\t7.9\n";

const PAIRS: &str = "Target\tRegulator\nG1\tR1\nG1\tR2\nG2\tR1\nG2\tR2\n";

const CLEAN_REGS: &str = "RegID\tS1\tS2\tS3\tS4\tS5\tS6\n\
                          R1\t1.0\t1.1\t0.9\t6.0\t6.1\t5.9\n\
                          R2\t2.0\t2.4\t1.8\t9.0\t9.3\t8.7\n";

const TWO_GROUPS: &str = "Sample\tCtrl\tTrt\n\
                          S1\t1\t0\nS2\t1\t0\nS3\t1\t0\n\
                          S4\t0\t1\nS5\t0\t1\nS6\t0\t1\n";

/// Every sample in one condition. MORE fits `target ~ Group + Group:Regulator`,
/// so there is no contrast to estimate. The two methods did not even agree on
/// what to do with it: PLS1 wrote an empty table, MLR wrote 887 rows on the
/// fixture this was found on. Both exited 0.
#[test]
fn a_design_with_one_condition_group_is_refused() {
    let one_group = "Sample\tOnly\nS1\t1\nS2\t1\nS3\t1\nS4\t1\nS5\t1\nS6\t1\n";
    for method in ["PLS1", "MLR"] {
        let dir = std::env::temp_dir()
            .join(format!("more_rs_onegroup_{method}_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();

        let run = run_more(&dir, one_group, TWO_TARGETS, CLEAN_REGS, PAIRS, method);
        let said = String::from_utf8_lossy(&run.stderr).into_owned()
            + &String::from_utf8_lossy(&run.stdout);
        assert!(!run.status.success(), "{method} accepted a one-group design:\n{said}");
        assert!(
            said.contains("only one condition group"),
            "{method} must say what is wrong, got:\n{said}"
        );
        // Silence is the actual defect: an exit code nobody reads is no better
        // than the empty table it replaced.
        assert!(said.contains("at least two groups"), "{method} must say what to do:\n{said}");
        fs::remove_dir_all(&dir).ok();
    }
}

/// The same design, still two groups, but every regulator carries a missing
/// value — and it survives both filters, which is the whole reason the defect
/// is reachable.
///
/// The two `PERC_NA` filters are independent: a regulator goes if more than
/// 20% of its samples are NA, and a sample goes if more than 20% of its
/// regulators are. One NA per regulator, spread round-robin across the
/// samples, passes both — 1 of 8 samples is 12.5% per regulator, and 2 of 10
/// regulators is exactly 20% per sample, which is not *more* than 20%. So
/// every NA reaches the design matrix, and from there no target can be fitted
/// at all.
///
/// A smaller fixture does not reproduce it. With two regulators the sample
/// filter fires first (one NA is 50% of that column), drops the affected
/// samples, and the run succeeds on what is left.
#[test]
fn a_run_where_no_target_could_be_fitted_is_refused() {
    const SAMPLES: usize = 8;
    const REGS: usize = 10;
    const TARGETS: usize = 3;

    let names: Vec<String> = (0..SAMPLES).map(|i| format!("S{i}")).collect();
    let header = |first: &str| format!("{first}\t{}\n", names.join("\t"));

    // Half the samples in each condition, so the design itself is sound.
    let mut design = String::from("Sample\tCtrl\tTrt\n");
    for (i, s) in names.iter().enumerate() {
        design.push_str(&format!("{s}\t{}\t{}\n", i32::from(i < SAMPLES / 2), i32::from(i >= SAMPLES / 2)));
    }

    let mut targets = header("GeneID");
    for t in 0..TARGETS {
        targets.push_str(&format!("G{t}"));
        for i in 0..SAMPLES {
            targets.push_str(&format!("\t{:.3}", (i as f64) * 1.7 + (t as f64)));
        }
        targets.push('\n');
    }

    let mut regs = header("RegID");
    let mut assoc = String::from("Target\tRegulator\n");
    for r in 0..REGS {
        regs.push_str(&format!("R{r}"));
        for i in 0..SAMPLES {
            // One NA per regulator, walked across the samples so no single
            // sample crosses the filter either.
            if i == r % SAMPLES {
                regs.push_str("\tNA");
            } else {
                regs.push_str(&format!("\t{:.3}", (i as f64) * 0.9 + (r as f64) * 0.4));
            }
        }
        regs.push('\n');
        for t in 0..TARGETS {
            assoc.push_str(&format!("G{t}\tR{r}\n"));
        }
    }

    for method in ["PLS1", "MLR"] {
        let dir =
            std::env::temp_dir().join(format!("more_rs_nofit_{method}_{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();

        let run = run_more(&dir, &design, &targets, &regs, &assoc, method);
        let said = String::from_utf8_lossy(&run.stderr).into_owned()
            + &String::from_utf8_lossy(&run.stdout);

        // The fixture is only meaningful if the filters really did keep
        // everything; if a future threshold change eats the NAs first, this
        // test must fail loudly rather than pass for the wrong reason.
        assert!(
            said.contains(&format!("{REGS} regulators kept, 0 dropped for missing values")),
            "the fixture must reach the model with its NAs intact, got:\n{said}"
        );
        assert!(
            !said.contains("Number of observations with missing values"),
            "the sample filter must not fire, or the NAs never reach the design:\n{said}"
        );

        assert!(
            !run.status.success(),
            "{method} reported success having modelled nothing:\n{said}"
        );
        assert!(
            said.contains(&format!("None of the {TARGETS} target features produced a model")),
            "{method} must count what failed, got:\n{said}"
        );
        // The per-target reason is the only thing that distinguishes this from
        // a dataset with no signal, so it has to survive into the message.
        assert!(
            said.contains(&format!("{TARGETS} x ")),
            "{method} must carry the per-target reason, got:\n{said}"
        );
        fs::remove_dir_all(&dir).ok();
    }
}

/// The guard must not fire on a dataset that simply has nothing to report.
/// A fit that finds no significant regulator is a legitimate, successful
/// answer, and conflating the two would replace a silent empty result with a
/// spurious error — the same defect wearing the other hat.
#[test]
fn a_dataset_with_no_signal_still_succeeds() {
    let dir = std::env::temp_dir().join(format!("more_rs_nosignal_{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();

    // Regulators uncorrelated with either target, and no missing values.
    let noise = "RegID\tS1\tS2\tS3\tS4\tS5\tS6\n\
                 R1\t0.4\t9.1\t0.2\t8.8\t0.5\t9.4\n\
                 R2\t7.7\t0.3\t7.2\t0.6\t7.9\t0.1\n";
    let run = run_more(&dir, TWO_GROUPS, TWO_TARGETS, noise, PAIRS, "PLS1");
    let said = String::from_utf8_lossy(&run.stderr).into_owned()
        + &String::from_utf8_lossy(&run.stdout);
    assert!(run.status.success(), "a signal-free dataset is not an error:\n{said}");
    assert!(
        said.contains("target features produced a model"),
        "the count is reported either way, got:\n{said}"
    );
    fs::remove_dir_all(&dir).ok();
}
