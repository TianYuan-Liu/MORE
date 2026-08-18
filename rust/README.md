# more-rs — MORE's modelling kernel in Rust

A reimplementation of the MORE R package's per-target modelling kernel, drop-in
at the seam PaintOmics uses (`PaintomicsServer/src/common/bioscripts/runMORE.R`):
same command-line options, same output files, same names.

It exists for two reasons. It is much faster — PLS1 by a few hundred times, MLR
by roughly an order of magnitude — and it is a single static binary, so it runs
on deployment images that have neither R nor the MORE package installed. On
`paintomics.uv.es` that is the actual situation: R is present but MORE is not,
so the R engines cannot run there at all and the port is the only way to run
either method.

| | implemented | agreement with R |
| --- | --- | --- |
| PLS1 (`varSel="Jack"`) | yes | **byte-identical** output files |
| MLR (`varSel="EN"`) | yes | every *decision* identical; coefficients differ — see below |
| ISGL, PLS2, `varSel="Perm"`, plotting | no | out of scope |

---

## The part worth understanding: how MLR reproduces R

MLR was the hard one, and the approach is the interesting bit.

### The problem

MORE's MLR path draws from R's random number generator. Two different jobs on
the same data give different answers under different seeds, because those draws
decide real things: which member of a group of correlated regulators becomes
that group's representative, and which observations land in which
cross-validation fold. For years the working assumption in this port was that
matching R therefore required "reproducing R's Mersenne-Twister stream", and
that this was out of reach.

### The approach

It is not out of reach. It is about sixty lines of Rust.

`src/rrng.rs` transcribes R 4.6.0's own C, rather than reaching for a
Mersenne-Twister crate — because the details that matter are not in the
algorithm, they are in R's wrapping of it:

* `do_setseed`'s 50-round LCG scramble of the seed, then `RNG_Init`'s 625-word
  fill, then `FixupSeeds` — miss the discarded first word and every draw is wrong;
* `MT_genrand` with R's tempering constants, behind `fixup()`, which keeps the
  result strictly inside (0, 1);
* `R_unif_index` under `sample.kind = "Rejection"` (R ≥ 3.6.0), including
  `rbits`' loop condition `n <= bits` — so `R_unif_index(1)` still *consumes* a
  draw even though its answer can only be 0. Two streams that disagree about
  that desynchronise permanently;
* `do_sample`'s without-replacement loop, so `sample(x, 1)` and a full
  permutation both advance the stream exactly as R advances it.

The result reproduces `runif()` to the bit and `sample()` exactly.

### Finding out where R actually draws

This is the part that does not work by reading the source.

A plain `grep 'sample(' R/` that also filters comment lines **misses two of the
three call sites**, because `MORE_MLR.R:811` and `:860` each carry a trailing
Spanish comment on the same line. Anyone auditing this path from grep output
alone concludes there is one draw where there are three.

So the call sites were established by instrumenting a *live run*.
`equivalence/rng/trace_shim.R` unlocks `base::sample` in `baseenv()` and
installs a pass-through recorder, then `runMORE.R` is run unmodified on top of
it. Every draw is logged in order, with its input vector and its result:

| site | R | when |
| --- | --- | --- |
| pair representative | `MORE_MLR.R:811` | exactly one correlated pair, once |
| clique representative | `MORE_MLR.R:860` | once per **complete** component |
| star-peel tie-break | `MORE_MLR.R:922` | only on a degree **and** sum tie |
| `cv.glmnet` folds | `foldid = sample(rep(seq(nfolds), length = N))` | 11 per target, one per alpha |

`CollinearityFilter2`'s three draws are unreachable — `GetMLR` passes
`col.filter = 'cor'`. `p.coef.pls2`'s is the `varSel = "Perm"` path `runMORE.R`
never takes. PLS1 reaches none, which is why it was already byte-exact.

### Three things that had to be right first

Reproducing the *stream* is not enough if you consume it at different moments.
Each of these was found by dumping live intermediates from both sides:

1. **`CollinearityFilter1` correlates the scaled matrix**, `scale(data, ...)`,
   not the raw values. Pearson is scale-invariant, so the two agree to about 15
   digits — and that is not enough, because R's tie-break asks
   `which(sums == max(sums))`, *exact* float equality, and whether that returns
   one index or two decides **whether R draws at all**. A last-bit disagreement
   does not perturb the answer slightly; it desynchronises everything after it.
2. **igraph numbers vertices by first appearance** in the edge table, and
   component ids follow that numbering. That order is the vector `sample()`
   indexes into.
3. **`CollinearityFilter1` appends** a new `reg.table` row for a group
   representative rather than renaming in place, so R's design column order is
   *[regulators never grouped]* followed by *[representatives]*. Same columns,
   different order — which matters because the solver is order-sensitive at
   MORE's tolerance (below).

### Keeping the fan-out

One stream has to be consumed in one order, which would force targets to be
fitted sequentially — and single-threaded, the port did not finish a
957-target job inside ten minutes, i.e. *slower than R*.

So `fit_all` splits the pass. Phase 1 walks the targets in R's order and takes
**only the draws**; phase 2 fans the elastic net out over rayon with the draws
already in hand. Phase 2 touches the stream not at all, so the fan-out cannot
perturb it — confirmed by the trace still matching after the split.

### What this achieves, measured

| dataset | RNG draws | mismatches | representatives wrong |
| --- | --- | --- | --- |
| 12 × 20 × 20 probe | 144 | **0** | 0 |
| `06-regulatory-more` (simulated, two omics) | 2830 | **0** | 0 of 766 |
| `11-stategra-more` (real, 957 targets) | 8157 | **0** | 0 of 1555 |

---

## What is *not* reproduced, and why that is R's property

Coefficient **values** differ, and no implementation can fix that.

MORE runs glmnet at `epsilon = 1e-5`, where coordinate descent has not
converged. glmnet's own objective on one fit, as its tolerance tightens:

| `thres` | objective | ‖b‖₁ |
| --- | --- | --- |
| **1e-5 — MORE's value** | 2.682616e-02 | 9.646 |
| 1e-8 | 2.641452e-02 | 10.148 |
| 1e-12 | 2.640745e-02 | 10.164 |
| 1e-14 | 2.640738e-02 | 10.164 |

Worse, at 1e-5 its answer is **not a function of the data**. Permuting the
design columns cannot move the optimum, yet it moves the result — objective
spread 3.0e-03 relative, against 5.4e-07 once converged.

So "reproduce R's coefficients exactly" is ill-posed: R does not reproduce
itself under a relabelling of its own inputs. The reachable bar is the one
already used for stochastic paths — *inside R's own measured spread*. Comparing
this port's gap to R against R's gap to itself under that permutation:

| design | R vs R (median / worst) | port vs R |
| --- | --- | --- |
| des_01 | 4.16e-03 / 8.30e-03 | 7.55e-03 |
| des_03 | 6.91e-03 / 1.01e-02 | 9.67e-03 |
| des_10 | 5.45e-03 / 1.21e-02 | **5.22e-03** |

Inside the band in all three, below R's median in one. End to end on the real
957-target dataset that comes out as edge Jaccard **0.991** (14 of ~1576 pairs
differ, precision 0.994), with the ID-only output files byte-identical.

**This is why the PaintOmics engine `rust-mlr` is opt-in and never substituted
silently.** PLS1 earned a silent default by being byte-identical; MLR has not,
so `auto`, stored jobs and older clients all keep getting R.

`DESCENT_THRESH` is MORE's own `1e-5` for the same reason — copying R's
tolerance is better than a tuned one now that the stream matches. Measured on
the real dataset: Jaccard 0.9911 against 0.9879 at 1e-7, 87/719 targets with an
identical R² against 7/719, worst coefficient 2.11 against 32.6, and 9× faster.

---

## Building

```sh
cargo build --release                     # native
cargo test  --release                     # 149 tests, no R required
```

For a Linux deployment target, cross-compiling from macOS needs **no C
toolchain** — the crate is pure Rust, so `rust-lld` links it:

```sh
rustup target add x86_64-unknown-linux-musl
cargo build --release --target x86_64-unknown-linux-musl \
  --config 'target.x86_64-unknown-linux-musl.linker="rust-lld"'
```

That produces a `static-pie` x86-64 binary needing nothing from the host.

> **Know before you ship it:** the musl build's MLR is markedly slower than the
> native one, because the allocation-heavy inner loops meet musl's allocator
> across rayon threads. Measured on a 6-vCPU VM against the same 957-target
> dataset: PLS1 3 s (about 3× the developer machine) but MLR 571 s (about 23×).
> PaintOmics' runtime guard is calibrated on a developer machine, so on such a
> host set `PAINTOMICS_MORE_COST_SCALE` or it will wave through jobs that then
> hit the queue timeout.

Drop the binary at `PaintomicsServer/src/common/bioscripts/more-rs`, or point
`PAINTOMICS_MORE_RS` at it.

## Verifying against R

Comparing output files alone cannot tell "same answer" from "same answer by
luck". The gate for the MLR path is that **the RNG streams match row for row**:

```sh
# 1. a traced R run, through the real product seam
cat equivalence/rng/trace_shim.R  path/to/runMORE.R > /tmp/runMORE_traced.R
MORE_R_RNG_TRACE=/tmp/r.tsv Rscript /tmp/runMORE_traced.R --method MLR ... --output_dir /tmp/r

# 2. the port, same arguments
MORE_RS_RNG_TRACE=/tmp/p.tsv ./target/release/more-rs --method MLR ... --output_dir /tmp/p

# 3. streams must be identical before any output claim is believed
diff <(cut -f2-5 /tmp/r.tsv) <(cut -f2-5 /tmp/p.tsv) && echo "streams identical"
```

The full equivalence harness — 9 parameter sets, R and the port on generated
data, scored on edges and coefficients — is:

```sh
python3 equivalence/run_equivalence.py
```

`KNOWN_MLR_DIVERGENCE` in that script pins the one remaining difference (3 of
190 edges on one configuration, precision 1.0000) so that it fails the run if it
grows.

## Where to read further

* **`BUILD.md`** — every environment hook (`MORE_RS_RNG_TRACE`,
  `MORE_RS_DESCENT_THRESH`, the design/CV/elastic-net probes) and what each is for.
* **`SPEC.md`** — the reference specification, mapped line by line onto `../R/`.
  §4 is the MLR investigation record, kept including the mechanisms that
  measurement later falsified; **§4.13 supersedes it** and is the one to read.

A note on method, because it is the most transferable thing here: on this code
path, reading the R source produced plausible wrong answers at a high rate —
five mechanisms were proposed for one discrepancy and four were falsified — and
only dumping live intermediates from both implementations produced correct ones.
Instrument first.
