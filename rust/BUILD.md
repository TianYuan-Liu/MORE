# Building `more-rs`

## Native (development)

```sh
cd rust
cargo test           # 140 unit tests (incl. 7 against live ropls/MORE output)
                     # plus the end-to-end output-contract test in tests/
cargo build --release
```

## Static musl binary for `python:3.9-slim-bookworm`

```sh
rustup target add x86_64-unknown-linux-musl
RUSTFLAGS="-C linker=rust-lld -C target-feature=+crt-static" \
  cargo build --release --target x86_64-unknown-linux-musl
```

Produces `target/x86_64-unknown-linux-musl/release/more-rs` (~1.2 MB,
`static-pie linked`). No dynamic dependencies and no R runtime, so it drops
into the PaintOmics server image as a file.

This cross-compiles from macOS with no C toolchain and no `musl-gcc`, which is
a direct consequence of having **no BLAS dependency**: every dependency
(`clap`, `rayon`, `statrs`) is pure Rust. Phase-1 profiling measured BLAS at
0.1% of the R pipeline's on-CPU time and the per-target matrices are ~20 rows,
so there was nothing to gain from linking one and a portability cost to pay.

**Verification status.** The artifact is confirmed to be a static-pie x86-64
ELF by `file`. It has *not* been executed inside `python:3.9-slim-bookworm` —
Docker was unavailable in the environment where it was built. Run the smoke
test below before relying on it in the image.

```sh
docker run --rm --platform linux/amd64 \
  -v "$PWD/target/x86_64-unknown-linux-musl/release/more-rs:/more-rs:ro" \
  -v "$PWD/equivalence:/eq:ro" -v /tmp/out:/out \
  python:3.9-slim-bookworm /more-rs --help
```

## Wiring it in behind an env flag

**Applied** (2026-08-10) in `MOREServlet._resolveMOREBackend`. Set
`PAINTOMICS_MORE_RS` to the binary's absolute path; leave it unset and every
job shells out to `Rscript runMORE.R` exactly as before. The arguments are the
same either way, because the CLI surfaces match option for option.

R wins in three cases, each of which would otherwise be a silent failure:

* `--method MLR`, and any method the port does not recognise. The port covers
  PLS1 only and exits pointing back at `runMORE.R` rather than silently doing
  something different, so routing MLR to it would turn a working analysis into
  a failed one.
* A configured path that is not on disk — the binary ships separately from the
  server and is absent from the deploy image, so a stale setting must degrade
  rather than take MORE down.
* A configured path without the executable bit, which an unpacked archive
  loses easily.

`PaintomicsServer/src/tests/test_more_backend_selection.py` pins all three.

## Equivalence

```sh
python3 equivalence/run_equivalence.py          # 7 parameter sets
RUNMORE_R=/path/to/runMORE.R python3 equivalence/run_equivalence.py
```

Requires R with `MORE` and `optparse` installed. Writes a provenance record to
`equivalence/fixtures/equivalence_report.json`.

### Against the bundled PaintOmics example

The synthetic sets above are generated to exercise the kernel. The shipped
`06-regulatory-more` dataset (250 targets, two regulatory omics of 40
regulators each, 12 samples, 4 conditions, PLS1, `minVariation=NA`) is the
end-to-end check, and R is deterministic on it — two independent runs produced
byte-identical output for all seven files, so no seed band is needed.

Measured 2026-08-10, R 4.6.0 / MORE 1.0.1, on **two** configurations — with the
association files, and with `--assoc_files NULL`, which PaintOmics reaches
whenever an omic is submitted without associations:

| File | With associations | `--assoc_files NULL` |
| --- | --- | --- |
| `MORE_output_*` | byte-identical | byte-identical |
| `MORE_relevant_assoc_*` | byte-identical | byte-identical |
| `MORE_relevant_pairs_*` | byte-identical | byte-identical |
| `MORE_rpc_*` | same rows, **different order** | byte-identical |

The rpc rows are identical as a multiset — every value, including sign, agrees
to the byte. They differ only in which omic comes first within a target: MORE
orders them by omic name under R's collation (`miRNA-seq` before
`Transcription_factor`, independent of `--omic_names` order — verified by
swapping the declaration order and getting the same output order), while the
port emits them in declaration order. That collation is locale-dependent —
`LC_COLLATE=C` would flip R's own order — so it is deliberately not
reproduced. `PathwayAcquisitionJob` reads the file into a dict-per-row for the
Step-3 panel, where the only order-sensitive behaviour is the 100 000-row
`df.head` cap, and that cap already truncates an arbitrary order under R.

Those runs also **found three real divergences**, all since fixed. None was
caught by the synthetic sets above, because each needs a condition the
generator never produces:

1. **The values and association files were built from the *modelling* matrix**
   rather than the input matrix, so every pair whose regulator had been dropped
   by the high-NA or low-variation filter went missing — 94 of 750 TF pairs and
   36 of 750 miRNA pairs. R builds them from `regulatoryData[[name]]`, which its
   regulator filters never touch because they run inside MORE on MORE's own
   copy. `prep::Omic` now carries `input_data` alongside `data` for this.
2. **`--assoc_files NULL` produced an empty values file.** With no association
   file there is no input pair set to snapshot, and R falls back to MORE's own
   significant pairs; the port returned nothing, writing a values file with only
   a header while reporting 5313 significant pairs. `full_pairs` now takes the
   significant set as that fallback.
3. **`format_r_double` modelled R's fixed/scientific switch as a threshold**
   (`e < -5 || e >= 15`). R has no such threshold: it renders both forms at 15
   significant digits and keeps the shorter, preferring fixed on a tie. So R
   writes `7.255e-05` where the port wrote `0.00007255`, and `1e+05` where it
   wrote `100000`. Values parse identically either way — the consumer runs
   `pd.to_numeric` — but it broke byte-comparison on any table containing small
   coefficients, which is every table where all regulators enter the model.

`tests/values_file_carries_every_input_pair.rs` guards (1) and (2) end-to-end
through the real binary; (3) is pinned by a unit test carrying the R oracle
values.

## Environment hooks

None of these change what a production run does; they exist so the port can be
compared against R rather than trusted.

| variable | effect |
| --- | --- |
| `MORE_RS_RNG_TRACE=<file>` | log every RNG draw, in stream order, in the same shape `equivalence/rng/trace_shim.R` logs for R. Diffing the two is the acceptance gate for the MLR path — comparing outputs alone cannot tell "same answer" from "same answer by luck". |
| `MORE_RS_DESCENT_THRESH=<f>` | override the coordinate-descent tolerance (default `1e-5`, MORE's own `epsilon`). Needed to compare against R with **both** sides converged, because at MORE's default glmnet is not. |
| `MORE_RS_DEBUG_EDGES=1` | dump `myreg` order, the `mycor` edge list with correlations, and every star-peel decision (max degree, candidate set, summed `abs(r)`, the tie). |
| `MORE_RS_DEBUG_MLR=1` | per-target groups, chosen `(alpha, lambda)`, non-zero count, deviance ratio. |
| `MORE_RS_DEBUG_DESIGN=<dir>` | write each target's design matrix, response first, in the shape `equivalence/design_probe.R` dumps from R. |
| `MORE_RS_CV_CURVE=<alpha>` | with `MORE_RS_EN_PROBE`, print every rung of the CV curve (`lambda`, `cvm`, `cvsd`, non-zero) for one alpha. Compare against `equivalence/cv_curve.R`, which prints `cv.glmnet`'s own. |
| `MORE_RS_EN_PROBE=<tsv>` | run only the elastic net, on a design matrix dumped from R. |
| `MORE_RS_EN_PATH=<alpha>` / `MORE_RS_EN_BETA=<rung>` / `MORE_RS_EN_THRESH=<f>` | lambda path, coefficients at one rung, tolerance for the probe. |
| `MORE_RS_GROUPS_OVERRIDE=<dir>` | replay a collinearity grouping captured from R instead of computing one. **Bypasses the RNG draws, so a run using it is not stream-equivalent.** |

### Reproducing the R comparison

```sh
# 1. a traced R run through the real product seam
cat equivalence/rng/trace_shim.R \
    ../../paintomics4/PaintomicsServer/src/common/bioscripts/runMORE.R > /tmp/runMORE_traced.R
MORE_R_RNG_TRACE=/tmp/r_trace.tsv Rscript /tmp/runMORE_traced.R --method MLR ... --output_dir /tmp/r

# 2. the port, same arguments
MORE_RS_RNG_TRACE=/tmp/p_trace.tsv ./target/release/more-rs --method MLR ... --output_dir /tmp/p

# 3. the streams must match row for row before any output claim is believed
diff <(cut -f2-5 /tmp/r_trace.tsv) <(cut -f2-5 /tmp/p_trace.tsv) && echo "streams identical"
```

`MORE_R_EPSILON` on the R side (also provided by `trace_shim.R`) forces the
tolerance glmnet is given, which is the only way to compare with both sides
converged.
