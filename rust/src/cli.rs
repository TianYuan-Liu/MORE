//! Command-line surface, mirroring `runMORE.R`'s `optparse` options exactly.
//!
//! Short forms exist only where the R script defines them (`-t -c -o -d -a -m`);
//! `--alpha`, `--vip`, `--filter_r2`, `--min_variation`, `--output_dir` and
//! `--date_seed` are long-only there and must stay long-only here. Defaults
//! match `runMORE.R:25-38` value for value, because `MOREServlet` relies on
//! several of them being omitted.

use clap::Parser;

/// Per-omic low-variation threshold. `NA` (or any non-numeric token) selects
/// MORE's automatic threshold: 10% of the maximum variability observed across
/// conditions.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MinVariation {
    Auto,
    Value(f64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Pls1,
    Mlr,
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Method::Pls1 => "PLS1",
            Method::Mlr => "MLR",
        })
    }
}

#[derive(Parser, Debug)]
#[command(name = "more-rs", about = "MORE regulatory analysis (Rust port)")]
struct Raw {
    #[arg(short = 't', long = "target_file")]
    target_file: String,
    #[arg(short = 'c', long = "condition_file")]
    condition_file: String,
    #[arg(short = 'o', long = "omic_names")]
    omic_names: String,
    #[arg(short = 'd', long = "data_files")]
    data_files: String,
    #[arg(short = 'a', long = "assoc_files")]
    assoc_files: String,
    #[arg(long = "min_variation", default_value = "0")]
    min_variation: String,
    #[arg(short = 'm', long = "method", default_value = "PLS1")]
    method: String,
    #[arg(long = "alpha", default_value_t = 0.05)]
    alpha: f64,
    #[arg(long = "vip", default_value_t = 0.8)]
    vip: f64,
    #[arg(long = "filter_r2", default_value_t = 0.0)]
    filter_r2: f64,
    #[arg(long = "output_dir")]
    output_dir: String,
    #[arg(long = "date_seed", default_value = "results")]
    date_seed: String,
    /// Drop the condition x regulator interaction terms.
    ///
    /// Not part of `runMORE.R`'s surface — PaintOmics never passes it, so the
    /// drop-in contract is unaffected. It exists because MORE's `interactions`
    /// argument defaults to TRUE and `runMORE.R` never overrides it, which
    /// makes interactions load-bearing: `RegulationPerCondition` reads
    /// `Group_*:regulator` terms back out to build the per-condition table, so
    /// with this flag set `MORE_rpc_*.tab` has no per-condition coefficients.
    #[arg(long = "no_interactions", default_value_t = false)]
    no_interactions: bool,
}

/// Validated options. One entry per omic in `omic_names` order, positionally
/// aligned across `data_files`, `assoc_files` and `min_variation`.
#[derive(Clone, Debug)]
pub struct Options {
    pub target_file: String,
    pub condition_file: String,
    /// Sanitised: spaces replaced with underscores, as `runMORE.R:131` does.
    pub omic_names: Vec<String>,
    pub data_files: Vec<String>,
    /// `None` where the R script would see the literal string `NULL`.
    pub assoc_files: Vec<Option<String>>,
    pub min_variation: Vec<MinVariation>,
    pub method: Method,
    pub alpha: f64,
    pub vip: f64,
    pub filter_r2: f64,
    pub output_dir: String,
    pub date_seed: String,
    pub interactions: bool,
}

impl Options {
    pub fn parse_args() -> Result<Options, String> {
        Options::from_raw(Raw::parse())
    }

    fn from_raw(raw: Raw) -> Result<Options, String> {
        let omic_names: Vec<String> = split_list(&raw.omic_names)
            .into_iter()
            .map(|s| s.trim().replace(' ', "_"))
            .collect();
        if omic_names.is_empty() || omic_names.iter().any(|s| s.is_empty()) {
            return Err("--omic_names must be a comma-separated list of non-empty names".into());
        }

        let data_files = split_list(&raw.data_files);
        if data_files.len() != omic_names.len() {
            return Err(format!(
                "--data_files has {} entries but --omic_names has {}; they are positionally aligned",
                data_files.len(),
                omic_names.len()
            ));
        }

        let assoc_files: Vec<Option<String>> = split_list(&raw.assoc_files)
            .into_iter()
            .map(|s| if s == "NULL" { None } else { Some(s) })
            .collect();
        if assoc_files.len() != omic_names.len() {
            return Err(format!(
                "--assoc_files has {} entries but --omic_names has {}; use the literal NULL for an omic without associations",
                assoc_files.len(),
                omic_names.len()
            ));
        }

        let min_variation = parse_min_variation(&raw.min_variation, omic_names.len());

        let method = match raw.method.as_str() {
            "PLS1" => Method::Pls1,
            "MLR" => Method::Mlr,
            other => {
                return Err(format!(
                    "--method must be PLS1 or MLR, got '{other}'"
                ))
            }
        };

        Ok(Options {
            target_file: raw.target_file,
            condition_file: raw.condition_file,
            omic_names,
            data_files,
            assoc_files,
            min_variation,
            method,
            alpha: raw.alpha,
            vip: raw.vip,
            filter_r2: raw.filter_r2,
            output_dir: raw.output_dir,
            date_seed: raw.date_seed,
            interactions: !raw.no_interactions,
        })
    }
}

fn split_list(raw: &str) -> Vec<String> {
    if raw.is_empty() {
        return Vec::new();
    }
    raw.split(',').map(|s| s.trim().to_string()).collect()
}

/// Ports `parse_min_variation` (`runMORE.R:318`): one token per omic, a single
/// token recycled to all omics, and any other count falling back to 0 for every
/// omic rather than risking a silent mis-alignment between thresholds and omics.
fn parse_min_variation(raw: &str, n_omics: usize) -> Vec<MinVariation> {
    let tokens = split_list(raw);
    let parsed: Vec<MinVariation> = tokens
        .iter()
        .map(|t| match t.parse::<f64>() {
            Ok(v) => MinVariation::Value(v),
            Err(_) => MinVariation::Auto,
        })
        .collect();

    if parsed.len() == 1 && n_omics > 1 {
        return vec![parsed[0]; n_omics];
    }
    if parsed.len() != n_omics {
        return vec![MinVariation::Value(0.0); n_omics];
    }
    parsed
}

#[cfg(test)]
mod tests {
    use super::*;

    fn raw() -> Raw {
        Raw {
            target_file: "t.tab".into(),
            condition_file: "c.tab".into(),
            omic_names: "TF,miRNA".into(),
            data_files: "tf.tab,mi.tab".into(),
            assoc_files: "tfa.tab,NULL".into(),
            min_variation: "0".into(),
            method: "PLS1".into(),
            alpha: 0.05,
            vip: 0.8,
            filter_r2: 0.0,
            output_dir: "out".into(),
            date_seed: "results".into(),
            no_interactions: false,
        }
    }

    #[test]
    fn the_literal_null_becomes_no_association_file() {
        let o = Options::from_raw(raw()).unwrap();
        assert_eq!(o.assoc_files[0].as_deref(), Some("tfa.tab"));
        assert_eq!(o.assoc_files[1], None);
    }

    #[test]
    fn spaces_in_omic_names_become_underscores() {
        let mut r = raw();
        r.omic_names = "DNase seq,miRNA".into();
        r.data_files = "a.tab,b.tab".into();
        r.assoc_files = "NULL,NULL".into();
        let o = Options::from_raw(r).unwrap();
        assert_eq!(o.omic_names[0], "DNase_seq");
    }

    #[test]
    fn a_data_file_count_mismatch_is_rejected() {
        let mut r = raw();
        r.data_files = "only_one.tab".into();
        let err = Options::from_raw(r).unwrap_err();
        assert!(err.contains("positionally aligned"), "{err}");
    }

    #[test]
    fn an_assoc_file_count_mismatch_is_rejected() {
        let mut r = raw();
        r.assoc_files = "one.tab".into();
        assert!(Options::from_raw(r).is_err());
    }

    #[test]
    fn an_unknown_method_is_rejected() {
        let mut r = raw();
        r.method = "PLS2".into();
        let err = Options::from_raw(r).unwrap_err();
        assert!(err.contains("PLS1 or MLR"), "{err}");
    }

    #[test]
    fn a_single_min_variation_token_is_recycled_to_every_omic() {
        let mut r = raw();
        r.min_variation = "0.3".into();
        let o = Options::from_raw(r).unwrap();
        assert_eq!(o.min_variation, vec![MinVariation::Value(0.3); 2]);
    }

    #[test]
    fn na_selects_the_automatic_threshold() {
        let mut r = raw();
        r.min_variation = "NA,0.5".into();
        let o = Options::from_raw(r).unwrap();
        assert_eq!(o.min_variation[0], MinVariation::Auto);
        assert_eq!(o.min_variation[1], MinVariation::Value(0.5));
    }

    #[test]
    fn a_mismatched_min_variation_count_falls_back_to_zero_for_all() {
        // R warns and uses 0 everywhere rather than mis-pairing thresholds.
        let mut r = raw();
        r.min_variation = "0.1,0.2,0.3".into();
        let o = Options::from_raw(r).unwrap();
        assert_eq!(o.min_variation, vec![MinVariation::Value(0.0); 2]);
    }

    #[test]
    fn an_empty_omic_name_is_rejected() {
        let mut r = raw();
        r.omic_names = "TF,".into();
        assert!(Options::from_raw(r).is_err());
    }

    #[test]
    fn defaults_match_the_r_option_list() {
        let o = Options::from_raw(raw()).unwrap();
        assert_eq!(o.method, Method::Pls1);
        assert_eq!(o.alpha, 0.05);
        assert_eq!(o.vip, 0.8);
        assert_eq!(o.filter_r2, 0.0);
        assert_eq!(o.date_seed, "results");
        assert!(o.interactions, "MORE's interactions default is TRUE");
    }

    #[test]
    fn interactions_can_be_switched_off_explicitly() {
        let mut r = raw();
        r.no_interactions = true;
        assert!(!Options::from_raw(r).unwrap().interactions);
    }
}
