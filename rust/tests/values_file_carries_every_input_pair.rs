//! The values and association files are an *unfiltered* snapshot of the input.
//!
//! `runMORE.R` builds them from `regulatoryData[[name]]`, which is the matrix as
//! read from disk (column-subset to the common samples and nothing more). Its
//! high-NA and low-variation filters run *inside* MORE, on MORE's own copy, and
//! never touch that object. So a regulator that MORE refuses to model still
//! contributes its `GENE:::REGULATOR` rows, and PA Step 1 shows every pair the
//! input declared — significance only drives the yellow-star overlay.
//!
//! This is an end-to-end test on purpose. The divergence it guards was not in
//! any single function: every unit was correct in isolation and `main` simply
//! handed the *modelling* matrix to the output path, which silently dropped 94
//! of 750 pairs on the bundled PaintOmics example.

use std::fs;
use std::path::PathBuf;
use std::process::Command;

fn write(dir: &PathBuf, name: &str, body: &str) -> String {
    let p = dir.join(name);
    fs::write(&p, body).unwrap();
    p.to_string_lossy().into_owned()
}

#[test]
fn a_regulator_dropped_for_low_variation_keeps_its_pairs_in_the_values_file() {
    let dir = std::env::temp_dir().join(format!("more_rs_unfiltered_{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let out = dir.join("out");

    // R2 is constant across every sample, so its sd across condition means is 0
    // and the automatic threshold (10% of the largest sd) drops it from the fit.
    let target = write(
        &dir,
        "target.tab",
        "GeneID\tS1\tS2\tS3\tS4\tS5\tS6\n\
         G1\t1.0\t1.2\t0.9\t5.0\t5.2\t4.8\n\
         G2\t2.0\t2.1\t1.9\t8.0\t8.1\t7.9\n",
    );
    let regs = write(
        &dir,
        "regs.tab",
        "RegID\tS1\tS2\tS3\tS4\tS5\tS6\n\
         R1\t1.0\t1.1\t0.9\t6.0\t6.1\t5.9\n\
         R2\t3.0\t3.0\t3.0\t3.0\t3.0\t3.0\n",
    );
    let assoc = write(
        &dir,
        "assoc.tab",
        "Target\tRegulator\nG1\tR1\nG1\tR2\nG2\tR1\nG2\tR2\n",
    );
    let design = write(
        &dir,
        "design.tab",
        "Sample\tCtrl\tTrt\n\
         S1\t1\t0\nS2\t1\t0\nS3\t1\t0\n\
         S4\t0\t1\nS5\t0\t1\nS6\t0\t1\n",
    );

    let run = Command::new(env!("CARGO_BIN_EXE_more-rs"))
        .args([
            "--target_file", &target,
            "--condition_file", &design,
            "--omic_names", "TF",
            "--data_files", &regs,
            "--assoc_files", &assoc,
            "--min_variation", "NA",
            "--method", "PLS1",
            "--output_dir", &out.to_string_lossy(),
            "--date_seed", "seed",
        ])
        .output()
        .expect("more-rs did not start");
    assert!(
        run.status.success(),
        "more-rs failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );

    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        stdout.contains("1 for low variation"),
        "the fixture must actually exercise the filter, got:\n{stdout}"
    );

    let values = fs::read_to_string(out.join("MORE_output_TF_seed.tab")).unwrap();
    let keys: Vec<&str> = values
        .lines()
        .skip(1)
        .map(|l| l.split('\t').next().unwrap())
        .collect();
    assert_eq!(
        keys,
        vec!["G1:::R1", "G1:::R2", "G2:::R1", "G2:::R2"],
        "every input pair belongs in the values file, including the dropped regulator's"
    );

    // The dropped regulator's row carries its raw values, exactly as read.
    let r2_row = values.lines().find(|l| l.starts_with("G1:::R2\t")).unwrap();
    assert_eq!(r2_row, "G1:::R2\t3\t3\t3\t3\t3\t3");

    let assoc_out = fs::read_to_string(out.join("MORE_relevant_assoc_TF_seed.tab")).unwrap();
    let mut assoc_lines: Vec<&str> = assoc_out.lines().collect();
    assoc_lines.sort_unstable();
    assert_eq!(assoc_lines, vec!["G1\tR1", "G1\tR2", "G2\tR1", "G2\tR2"]);

    // The yellow-star file stays significance-filtered: it must never gain a
    // regulator that was never modelled.
    let pairs = fs::read_to_string(out.join("MORE_relevant_pairs_TF_seed.tab")).unwrap();
    assert!(
        !pairs.contains("R2"),
        "a regulator that was never modelled cannot be significant, got:\n{pairs}"
    );

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn without_an_association_file_the_values_file_falls_back_to_the_significant_pairs() {
    // `--assoc_files NULL` is reachable from the product: MOREServlet's STEP1
    // leaves assoc_path None when the user uploads no association file, and
    // STEP2 passes the literal NULL. runMORE.R then has no input pair set to
    // snapshot, so it falls back to MORE's own significant pairs rather than
    // writing nothing -- measured against R 4.6.0 / MORE 1.0.1 on the bundled
    // 06-regulatory-more transcription-factor omic: 5313 significant pairs and
    // 5313 rows in the values file. Returning an empty set here hands PA Step 1
    // an omic with no GENE:::REGULATOR features at all.
    let dir = std::env::temp_dir().join(format!("more_rs_noassoc_{}", std::process::id()));
    fs::create_dir_all(&dir).unwrap();
    let out = dir.join("out");

    let target = write(
        &dir,
        "target.tab",
        "GeneID\tS1\tS2\tS3\tS4\tS5\tS6\n\
         G1\t1.0\t1.2\t0.9\t5.0\t5.2\t4.8\n\
         G2\t2.0\t2.1\t1.9\t8.0\t8.1\t7.9\n",
    );
    let regs = write(
        &dir,
        "regs.tab",
        "RegID\tS1\tS2\tS3\tS4\tS5\tS6\n\
         R1\t1.0\t1.1\t0.9\t6.0\t6.1\t5.9\n\
         R2\t2.0\t2.4\t1.7\t9.0\t9.3\t8.7\n",
    );
    let design = write(
        &dir,
        "design.tab",
        "Sample\tCtrl\tTrt\n\
         S1\t1\t0\nS2\t1\t0\nS3\t1\t0\n\
         S4\t0\t1\nS5\t0\t1\nS6\t0\t1\n",
    );

    let run = Command::new(env!("CARGO_BIN_EXE_more-rs"))
        .args([
            "--target_file", &target,
            "--condition_file", &design,
            "--omic_names", "TF",
            "--data_files", &regs,
            "--assoc_files", "NULL",
            "--min_variation", "NA",
            "--method", "PLS1",
            "--output_dir", &out.to_string_lossy(),
            "--date_seed", "seed",
        ])
        .output()
        .expect("more-rs did not start");
    assert!(
        run.status.success(),
        "more-rs failed: {}",
        String::from_utf8_lossy(&run.stderr)
    );

    let pairs: Vec<String> = fs::read_to_string(out.join("MORE_relevant_pairs_TF_seed.tab"))
        .unwrap()
        .lines()
        .map(|l| l.trim().trim_matches('"').to_string())
        .filter(|l| !l.is_empty())
        .collect();
    assert!(!pairs.is_empty(), "fixture produced no significant pairs to fall back to");

    let values = fs::read_to_string(out.join("MORE_output_TF_seed.tab")).unwrap();
    let keys: Vec<String> = values
        .lines()
        .skip(1)
        .map(|l| l.split('\t').next().unwrap().to_string())
        .collect();
    assert_eq!(
        keys, pairs,
        "with no association file the values file must carry the significant pairs"
    );

    let assoc_out = fs::read_to_string(out.join("MORE_relevant_assoc_TF_seed.tab")).unwrap();
    let assoc_keys: Vec<String> = assoc_out
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.replace('\t', ":::"))
        .collect();
    assert_eq!(assoc_keys, pairs, "the association file follows the same set");

    fs::remove_dir_all(&dir).ok();
}
