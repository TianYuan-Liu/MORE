# Building `more-rs`

## Native (development)

```sh
cd rust
cargo test           # 113 unit tests, incl. 7 against live ropls/MORE output
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

`runMORE.R` stays the fallback. The intended seam in `MOREServlet.py` is to
choose the binary when it is present and the flag is set, and to fall back to
`Rscript runMORE.R` otherwise — same arguments either way, since the CLI
surfaces match:

```python
more_bin = os.environ.get("PAINTOMICS_MORE_RS")
if more_bin and os.path.exists(more_bin):
    cmd = [more_bin] + args
else:
    cmd = ["Rscript", RUNMORE_R] + args
```

Not yet applied to `PaintomicsServer` — the port covers PLS1 only, and
`--method MLR` deliberately exits with a message pointing back at `runMORE.R`
rather than silently doing something different.

## Equivalence

```sh
python3 equivalence/run_equivalence.py          # 7 parameter sets
RUNMORE_R=/path/to/runMORE.R python3 equivalence/run_equivalence.py
```

Requires R with `MORE` and `optparse` installed. Writes a provenance record to
`equivalence/fixtures/equivalence_report.json`.
