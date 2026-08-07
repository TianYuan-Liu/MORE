//! Input parsing and sample alignment.
//!
//! Ports `read_matrix` and the loading half of `runMORE.R`. Two behaviours are
//! deliberate and must not be "simplified":
//!
//! * the separator is chosen by **what it produces**, not by whether the first
//!   attempt threw. `read.table(sep="\t")` on a comma-separated file succeeds
//!   with zero data columns, and the old code propagated that as an empty
//!   matrix, so a job ran to completion having modelled nothing;
//! * sample alignment is a strict name-based intersection. MORE's own R API
//!   falls back to positional alignment with a console warning; in a web
//!   context nobody sees that warning, and silently pairing differently-named
//!   samples by column order is how statistically meaningless results get
//!   published.

use std::collections::HashMap;
use std::fs;

/// A labelled numeric matrix: feature IDs down the rows, sample IDs across.
#[derive(Clone, Debug)]
pub struct Frame {
    pub row_names: Vec<String>,
    pub col_names: Vec<String>,
    /// `values[r][c]`; NaN carries R's `NA`.
    pub values: Vec<Vec<f64>>,
}

impl Frame {
    pub fn nrow(&self) -> usize {
        self.row_names.len()
    }
    pub fn ncol(&self) -> usize {
        self.col_names.len()
    }

    pub fn row_index(&self) -> HashMap<&str, usize> {
        self.row_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect()
    }

    /// Reorder columns to `wanted`, dropping the rest. Every name must exist.
    pub fn select_columns(&self, wanted: &[String]) -> Result<Frame, String> {
        let idx: HashMap<&str, usize> = self
            .col_names
            .iter()
            .enumerate()
            .map(|(i, n)| (n.as_str(), i))
            .collect();
        let picks: Vec<usize> = wanted
            .iter()
            .map(|w| idx.get(w.as_str()).copied().ok_or_else(|| format!("unknown sample '{w}'")))
            .collect::<Result<_, _>>()?;
        Ok(Frame {
            row_names: self.row_names.clone(),
            col_names: wanted.to_vec(),
            values: self
                .values
                .iter()
                .map(|row| picks.iter().map(|&c| row[c]).collect())
                .collect(),
        })
    }

    /// Keep only the named rows, in the order given.
    pub fn select_rows(&self, wanted: &[String]) -> Frame {
        let idx = self.row_index();
        let picks: Vec<usize> = wanted.iter().filter_map(|w| idx.get(w.as_str()).copied()).collect();
        Frame {
            row_names: picks.iter().map(|&r| self.row_names[r].clone()).collect(),
            col_names: self.col_names.clone(),
            values: picks.iter().map(|&r| self.values[r].clone()).collect(),
        }
    }
}

/// One parse attempt under a single separator.
struct Attempt {
    frame: Option<Frame>,
    problem: Option<String>,
}

fn parse_with(text: &str, sep: char) -> Attempt {
    let mut lines = text.lines().filter(|l| !l.is_empty());
    let header = match lines.next() {
        Some(h) => h,
        None => {
            return Attempt { frame: None, problem: Some("file is empty".into()) };
        }
    };

    // read.table(row.names = 1): the header names the data columns only, so a
    // header with one fewer field than the data rows is the normal shape. R
    // also accepts a header that includes a name for the ID column.
    let header_fields: Vec<&str> = header.split(sep).map(unquote).collect();

    let mut row_names = Vec::new();
    let mut values: Vec<Vec<f64>> = Vec::new();
    let mut width = None;
    for line in lines {
        let mut fields = line.split(sep);
        let id = match fields.next() {
            Some(f) => unquote(f).to_string(),
            None => continue,
        };
        let row: Vec<f64> = fields.map(|f| parse_cell(unquote(f))).collect();
        match width {
            None => width = Some(row.len()),
            Some(w) if w != row.len() => {
                return Attempt {
                    frame: None,
                    problem: Some(format!("row '{id}' has {} fields, expected {w}", row.len())),
                };
            }
            _ => {}
        }
        row_names.push(id);
        values.push(row);
    }

    let width = match width {
        // A header with nothing under it parses cleanly in R: `read.table`
        // returns a 0-row frame that still has columns, so the caller's
        // "no data rows" check is what rejects it. Reporting "no data columns"
        // here instead would misdiagnose the file.
        None => {
            let col_names = header_fields.iter().skip(1).map(|s| s.to_string()).collect::<Vec<_>>();
            return Attempt {
                frame: Some(Frame { row_names: Vec::new(), col_names, values: Vec::new() }),
                problem: None,
            };
        }
        Some(w) => w,
    };
    if width == 0 {
        return Attempt { frame: None, problem: Some("no data columns".into()) };
    }

    // R errors on duplicate row names rather than silently keeping the first.
    let mut seen = std::collections::HashSet::new();
    for name in &row_names {
        if !seen.insert(name.as_str()) {
            return Attempt {
                frame: None,
                problem: Some(format!("duplicate 'row.names' are not allowed: '{name}'")),
            };
        }
    }

    let col_names: Vec<String> = if header_fields.len() == width + 1 {
        header_fields[1..].iter().map(|s| s.to_string()).collect()
    } else if header_fields.len() == width {
        header_fields.iter().map(|s| s.to_string()).collect()
    } else {
        return Attempt {
            frame: None,
            problem: Some(format!(
                "header has {} fields but rows have {width} data columns",
                header_fields.len()
            )),
        };
    };

    Attempt { frame: Some(Frame { row_names, col_names, values }), problem: None }
}

fn unquote(s: &str) -> &str {
    s.trim().trim_matches(|c| c == '"' || c == '\'')
}

/// `NA`, `NaN` and empty become NaN, as R's `read.table` does for `NA`.
/// Anything else non-numeric is Infinity-tagged so the caller can name the
/// offending column rather than dying inside a model fit.
fn parse_cell(s: &str) -> f64 {
    if s.is_empty() || s == "NA" || s == "NaN" || s == "na" {
        return f64::NAN;
    }
    match s.parse::<f64>() {
        Ok(v) => v,
        // Sentinel: distinguishable from a legitimate NA, checked below.
        Err(_) => NON_NUMERIC,
    }
}

/// Sentinel for "this cell was not a number at all", as opposed to `NA`.
const NON_NUMERIC: f64 = -1.234_567_890_123_456_7e307;

fn is_non_numeric(v: f64) -> bool {
    v == NON_NUMERIC
}

/// Port of `read_matrix`. Tries tab then comma and keeps whichever yields more
/// data columns; rejects a parse with no data columns, no data rows, or any
/// genuinely non-numeric cell.
pub fn read_matrix(path: &str) -> Result<Frame, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("File does not exist or cannot be read at path: {path} ({e})"))?;

    let mut best: Option<Frame> = None;
    let mut problems = Vec::new();
    for (sep, label) in [('\t', "tab"), (',', "comma")] {
        let attempt = parse_with(&text, sep);
        if let Some(p) = attempt.problem {
            problems.push(format!("{label}: {p}"));
        }
        if let Some(f) = attempt.frame {
            if f.ncol() > 0 && best.as_ref().map_or(true, |b| f.ncol() > b.ncol()) {
                best = Some(f);
            }
        }
    }

    let frame = best.ok_or_else(|| {
        format!(
            "no data columns could be read from {path}. Tried tab and comma separators.{}",
            if problems.is_empty() {
                " Check the separator, and that feature IDs in the first column are unique.".to_string()
            } else {
                format!(" ({})", problems.join("; "))
            }
        )
    })?;

    if frame.nrow() == 0 {
        return Err(format!("{path} has a header but no data rows."));
    }

    let bad: Vec<String> = (0..frame.ncol())
        .filter(|&c| frame.values.iter().any(|row| is_non_numeric(row[c])))
        .map(|c| frame.col_names[c].clone())
        .collect();
    if !bad.is_empty() {
        return Err(format!(
            "{path} has non-numeric values in column(s): {}{}. Expected a numeric matrix with feature IDs in the first column.",
            bad.iter().take(10).cloned().collect::<Vec<_>>().join(", "),
            if bad.len() > 10 { format!(" (+{} more)", bad.len() - 10) } else { String::new() }
        ));
    }

    Ok(frame)
}

/// One (target, regulator, area) association row.
#[derive(Clone, Debug, PartialEq)]
pub struct Association {
    pub target: String,
    pub regulator: String,
    pub area: String,
}

/// Load and orient an association file.
///
/// MORE's contract is col1 = target, col2 = regulator, optional col3 = area.
/// Orientation is detected from columns 1-2 only, so a 3-column file keeps its
/// area column when a swap is needed. Four or more columns is an error rather
/// than a silent truncation.
pub fn read_associations(
    path: &str,
    omic: &str,
    regulator_ids: &[String],
    target_ids: &[String],
) -> Result<Vec<Association>, String> {
    let text = fs::read_to_string(path)
        .map_err(|e| format!("cannot read association file for omic '{omic}': {e}"))?;
    let mut rows: Vec<Vec<String>> = Vec::new();
    for (i, line) in text.lines().enumerate() {
        if line.is_empty() {
            continue;
        }
        let fields: Vec<String> = line.split('\t').map(|f| unquote(f).to_string()).collect();
        if i == 0 {
            continue; // header, as read.table(header = TRUE)
        }
        rows.push(fields);
    }

    let ncol = rows.iter().map(|r| r.len()).max().unwrap_or(0);
    if ncol < 2 {
        return Err(format!(
            "Association file for omic '{omic}' has only {ncol} column(s). MORE requires 2 columns (target, regulator) with an optional 3rd column for interaction type."
        ));
    }
    if ncol > 3 {
        return Err(format!(
            "Association file for omic '{omic}' has {ncol} columns. MORE accepts at most 3 columns: target, regulator, and an optional interaction-type/area column. Please check the file."
        ));
    }

    let regs: std::collections::HashSet<&str> = regulator_ids.iter().map(|s| s.as_str()).collect();
    let col1 = rows.iter().filter(|r| regs.contains(r[0].as_str())).count();
    let col2 = rows.iter().filter(|r| r.len() > 1 && regs.contains(r[1].as_str())).count();

    if col1.max(col2) == 0 {
        return Err(format!(
            "Association file for omic '{omic}' shares no regulator IDs with its data file, in either column.\n  association col 1: {}\n  association col 2: {}\n  {omic} data file: {}\nCheck that both files use the same regulator identifiers, and that the association file has a header row (its first line is read as one).",
            preview(rows.iter().map(|r| r[0].as_str())),
            preview(rows.iter().filter(|r| r.len() > 1).map(|r| r[1].as_str())),
            preview(regulator_ids.iter().map(|s| s.as_str()))
        ));
    }

    let swap = col1 > col2;
    let out: Vec<Association> = rows
        .iter()
        .filter(|r| r.len() >= 2)
        .map(|r| {
            let (t, g) = if swap { (&r[1], &r[0]) } else { (&r[0], &r[1]) };
            Association {
                target: t.clone(),
                regulator: g.clone(),
                area: r.get(2).cloned().unwrap_or_default(),
            }
        })
        .collect();

    let targets: std::collections::HashSet<&str> = target_ids.iter().map(|s| s.as_str()).collect();
    if !out.iter().any(|a| targets.contains(a.target.as_str())) {
        return Err(format!(
            "Association file for omic '{omic}' shares no target IDs with the target expression file.\n  association targets: {}\n  expression features: {}\nBoth files must identify features the same way (same ID type, same case).",
            preview(out.iter().map(|a| a.target.as_str())),
            preview(target_ids.iter().map(|s| s.as_str()))
        ));
    }

    Ok(out)
}

fn preview<'a>(it: impl Iterator<Item = &'a str>) -> String {
    let mut seen = Vec::new();
    for v in it {
        if !seen.contains(&v) {
            seen.push(v);
        }
        if seen.len() == 3 {
            break;
        }
    }
    seen.join(", ")
}

/// Strict name-based intersection across every input, preserving the order the
/// target file uses. Positional fallback is intentionally absent.
pub fn common_samples(target: &Frame, condition: &Frame, regulatory: &[Frame]) -> Vec<String> {
    let mut keep: Vec<String> = target
        .col_names
        .iter()
        .filter(|s| condition.row_names.contains(s))
        .cloned()
        .collect();
    for reg in regulatory {
        keep.retain(|s| reg.col_names.contains(s));
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn tmp(name: &str, body: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("more_rs_{}_{}", std::process::id(), name));
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(body.as_bytes()).unwrap();
        p.to_string_lossy().into_owned()
    }

    #[test]
    fn a_tab_file_parses() {
        let p = tmp("tab.tab", "ID\tS1\tS2\nG1\t1.0\t2.0\nG2\t3.0\t4.0\n");
        let f = read_matrix(&p).unwrap();
        assert_eq!(f.row_names, vec!["G1", "G2"]);
        assert_eq!(f.col_names, vec!["S1", "S2"]);
        assert_eq!(f.values[1][1], 4.0);
    }

    #[test]
    fn a_comma_file_parses_rather_than_yielding_an_empty_matrix() {
        // The regression: a tab parse of this file "succeeds" with zero data
        // columns, and the old code ran the whole job on nothing.
        let p = tmp("comma.tab", "ID,S1,S2\nG1,1.0,2.0\nG2,3.0,4.0\n");
        let f = read_matrix(&p).unwrap();
        assert_eq!(f.ncol(), 2);
        assert_eq!(f.values[0][0], 1.0);
    }

    #[test]
    fn a_header_with_no_data_rows_is_rejected() {
        let p = tmp("empty.tab", "ID\tS1\tS2\n");
        let err = read_matrix(&p).unwrap_err();
        assert!(err.contains("no data rows"), "{err}");
    }

    #[test]
    fn duplicate_feature_ids_are_rejected_not_silently_merged() {
        let p = tmp("dup.tab", "ID\tS1\nG1\t1.0\nG1\t2.0\n");
        let err = read_matrix(&p).unwrap_err();
        assert!(err.contains("no data columns") || err.contains("duplicate"), "{err}");
    }

    #[test]
    fn a_non_numeric_cell_names_its_column() {
        let p = tmp("bad.tab", "ID\tS1\tS2\nG1\t1.0\toops\n");
        let err = read_matrix(&p).unwrap_err();
        assert!(err.contains("non-numeric"), "{err}");
        assert!(err.contains("S2"), "{err}");
    }

    #[test]
    fn na_is_read_as_a_missing_value_not_a_parse_failure() {
        let p = tmp("na.tab", "ID\tS1\tS2\nG1\t1.0\tNA\n");
        let f = read_matrix(&p).unwrap();
        assert!(f.values[0][1].is_nan());
    }

    #[test]
    fn a_missing_file_is_reported_by_path() {
        let err = read_matrix("/nonexistent/nope.tab").unwrap_err();
        assert!(err.contains("/nonexistent/nope.tab"), "{err}");
    }

    #[test]
    fn associations_swap_when_the_regulator_is_in_column_one() {
        let p = tmp("assoc_sw.tab", "Regulator\tTarget\nTF1\tG1\nTF2\tG2\n");
        let regs = vec!["TF1".to_string(), "TF2".to_string()];
        let targets = vec!["G1".to_string(), "G2".to_string()];
        let a = read_associations(&p, "TF", &regs, &targets).unwrap();
        assert_eq!(a[0].target, "G1");
        assert_eq!(a[0].regulator, "TF1");
    }

    #[test]
    fn associations_keep_the_area_column_through_a_swap() {
        let p = tmp("assoc_area.tab", "Regulator\tTarget\tArea\nTF1\tG1\tPROMOTER\n");
        let regs = vec!["TF1".to_string()];
        let targets = vec!["G1".to_string()];
        let a = read_associations(&p, "TF", &regs, &targets).unwrap();
        assert_eq!(a[0].area, "PROMOTER");
        assert_eq!(a[0].regulator, "TF1");
    }

    #[test]
    fn associations_matching_no_regulator_are_rejected() {
        let p = tmp("assoc_none.tab", "Target\tRegulator\nG1\tNOPE\n");
        let regs = vec!["TF1".to_string()];
        let targets = vec!["G1".to_string()];
        let err = read_associations(&p, "TF", &regs, &targets).unwrap_err();
        assert!(err.contains("shares no regulator IDs"), "{err}");
    }

    #[test]
    fn associations_matching_no_target_are_rejected() {
        let p = tmp("assoc_notarget.tab", "Target\tRegulator\nZZZ\tTF1\n");
        let regs = vec!["TF1".to_string()];
        let targets = vec!["G1".to_string()];
        let err = read_associations(&p, "TF", &regs, &targets).unwrap_err();
        assert!(err.contains("shares no target IDs"), "{err}");
    }

    #[test]
    fn a_four_column_association_file_is_rejected_not_truncated() {
        let p = tmp("assoc_4.tab", "T\tR\tA\tX\nG1\tTF1\tP\tjunk\n");
        let regs = vec!["TF1".to_string()];
        let targets = vec!["G1".to_string()];
        let err = read_associations(&p, "TF", &regs, &targets).unwrap_err();
        assert!(err.contains("at most 3 columns"), "{err}");
    }

    #[test]
    fn sample_alignment_intersects_by_name_and_keeps_target_order() {
        let t = Frame {
            row_names: vec!["G1".into()],
            col_names: vec!["S2".into(), "S1".into(), "S3".into()],
            values: vec![vec![1., 2., 3.]],
        };
        let c = Frame {
            row_names: vec!["S1".into(), "S2".into()],
            col_names: vec!["Ctrl".into()],
            values: vec![vec![1.], vec![0.]],
        };
        let r = Frame {
            row_names: vec!["TF1".into()],
            col_names: vec!["S1".into(), "S2".into()],
            values: vec![vec![1., 2.]],
        };
        assert_eq!(common_samples(&t, &c, &[r]), vec!["S2", "S1"]);
    }

    #[test]
    fn sample_alignment_returns_nothing_when_names_disagree() {
        let t = Frame {
            row_names: vec!["G1".into()],
            col_names: vec!["A".into()],
            values: vec![vec![1.]],
        };
        let c = Frame {
            row_names: vec!["B".into()],
            col_names: vec!["Ctrl".into()],
            values: vec![vec![1.]],
        };
        assert!(common_samples(&t, &c, &[]).is_empty());
        let _ = c;
    }
}
