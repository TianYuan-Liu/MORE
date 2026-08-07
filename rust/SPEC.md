# MORE Rust port — reference specification

Reference implementation: the R sources in `../R/`, at fork commit `9ae6635cebaefa38daab8295a209d4856793c97a`
(identical to upstream `BiostatOmics/MORE` v1.0.1 — the fork is a pristine copy, so one SHA
identifies both). The R sources are never edited; they are the oracle.

Line references below are into `../R/` unless prefixed `ropls:`, which means
`ropls` 1.44.0's `.coreF`, dumped for reading but not vendored.

## 0. Scope

In scope, because the PaintOmics seam reaches them:

- `more()` dispatch for `varSel="Jack", method="PLS1"` and `varSel="EN", method="MLR"`
- the whole per-target PLS1 path: `GetPLS` → `ResultsPerTargetF.i` → `p.valuejack`
- the MLR path: `GetMLR` → `ResultsPerTargetF.i.mlr` → `ElasticNet`
- filters: missing-value, low-variation, collinearity, constant-column
- `RegulationPerCondition`, `FilterRegulationPerCondition`
- **regulator × condition interactions** (see §1.1 — not optional)

Out of scope: `GetISGL`, PLS2, `clinic`/`clinicType`, `scaleType != "auto"`,
`varSel="Perm"` (`p.coef`), and everything in `output_analysis.R` that draws
(`plotMORE`, `networkMORE`, `oraMORE`, `gseaMORE`, `summaryPlot`, …).

### 0.1 What `runMORE.R` actually requests

`PaintomicsServer/src/common/bioscripts/runMORE.R` calls:

```r
more(targetData, regulatoryData, associations, condition,
     method = opt$method,          # "PLS1" or "MLR"
     varSel = varSel_val,          # "Jack" for PLS1, "EN" for MLR
     minVariation = minVariation_vec,
     alfa = opt$alpha, vip = opt$vip)
```

Everything else takes its `more()` default. The defaults that matter:

| argument | default | consequence |
| --- | --- | --- |
| `interactions` | `TRUE` | **condition×regulator terms are built** — see §1.1 |
| `percNA` | `0.2` | regulators >20% NA dropped; samples >20% NA dropped |
| `scaleType` | `"auto"` | centre and unit-variance scale, no block weighting |
| `omicType` | `NULL` | inferred per omic by `isBin()` |
| `correlation` | `0.7` | collinearity threshold (MLR path only) |
| `seed` | `123` | `set.seed(123)` — irrelevant on the PLS1 path, see §3.5 |
| `parallel` | `FALSE` | |

### 0.1.1 The `interactions` scope correction

The port brief said "skip … interactions". That is not compatible with being a
drop-in replacement at the `runMORE.R` seam. `interactions` defaults to `TRUE`,
`runMORE.R` never overrides it, and `runMORE.R` always supplies `condition`, so
`des.mat` is non-NULL and `RegulatorsInteractions` (`auxFunctions.R:351`) builds
a `Group_*:regulator` column for every (condition level, regulator) pair.

`RegulationPerCondition` then *reads those interaction terms back*
(`output_analysis.R:105-127`) to produce the per-condition coefficient table
that becomes `MORE_rpc_*.tab`. Without interactions there are no per-condition
coefficients and the primary output file is empty. Interactions are therefore
implemented.

## 1. Design matrix construction (per target feature)

For target `g`, from `ResultsPerTargetF.i` (`MORE_PLS.R:704`):

1. `GetAllReg(g, associations, regulatoryData)` → every regulator associated with
   `g`, tagged with its omic and `area`. Rows whose regulator is the literal
   `"No-regulator"` are dropped.
2. `RemovedRegulators` marks each regulator `Model`, `MissingValue` or
   `LowVariation` and returns the value matrix for the `Model` ones.
3. `RegulatorsInteractions(TRUE, …, method="pls")` splits regulators by omic and,
   per omic, appends `expcond:regulator` columns for every condition column
   `expcond` (`model.matrix(~ Group_a:`reg` + …)`, intercept dropped, then the
   original `des.mat` columns are removed again). Columns with `sd == 0` are
   dropped here.
4. Each omic block is centred/scaled (`scale(x, TRUE, TRUE)` under `scaleType="auto"`),
   then `Scaling.type` concatenates the blocks (no reweighting for `"auto"`).
5. `des.mat2 = [response, scale(des.mat), regulator+interaction columns]`.
6. Columns with `sd == 0` (NA-tolerant: `is.na(sd) | sd > 0` is *kept*) are dropped;
   the regulators behind them are marked `filter = "Constant"`.

Note step 6 keeps columns whose sd is `NA`. That is deliberate in the R and must
be reproduced: an all-NA column survives into the fit.

### 1.1 Name mangling that the port must reproduce

`GetPLS:120-158` rewrites regulator IDs before anything else:

- `:` → `-` (because `:` is the interaction separator)
- trailing `_R`, `_P`, `_N` → `-R`, `-P`, `-N`
- if an omic shares any ID with the target omic, **every** ID in that omic is
  prefixed `"<omic>-"`; likewise if two regulatory omics share any ID, both are prefixed

The same substitutions are applied to column 2 of that omic's association table.

**Known upstream defect, deliberately not reproduced.**
`RegulationPerCondition` finishes with an unanchored global
`gsub(paste0(names(omicType), "-", collapse="|"), "", regulator)`, which deletes
`<omic>-` anywhere in an ID, so a genuine regulator `TF-1` under an omic named
`TF` is emitted as `1`. `runMORE.R` works around this after the fact
(`restore_regulator_ids`). The port emits the true IDs directly. The equivalence
harness must treat this as a **documented divergence**, comparing against the
R output *after* `runMORE.R`'s repair, not against raw `RegulationPerCondition`.

## 2. Global pre-filters (whole-run, before any target is fitted)

In `GetPLS` order:

1. rows of `targetData` with any `Inf`/`-Inf` → dropped, recorded `-Inf/Inf values`
2. regulators with any `Inf`/`-Inf` → dropped
3. targets with fewer than `ncol(targetData)` non-NA values → dropped,
   recorded `Too many missing values` (`min.obs = ncol(targetData)`, i.e. **any** NA drops the target)
4. targets absent from every association table → dropped, `TargetFNOregu`
5. targets with `sd == 0` → dropped, `Response values are constant`
6. regulators with `>percNA` missing → dropped (`myregNA`)
7. samples with `>percNA` missing in any omic → dropped from **all** matrices
8. `LowVariationRegu` → `myregLV`

Steps 6-8 are computed **once per `more()` call over all targets**. This is why
splitting a run into chunks is not semantically free: the auto (`NA`)
`minVariation` threshold is 10% of the maximum variability observed across the
whole call, so chunking changes which regulators survive. Any chunking mitigation
in `runMORE.R` must be validated by the equivalence harness, not assumed.

## 3. PLS1 (`method="PLS1"`, `varSel="Jack"`)

MORE calls `ropls::opls(X, y, scaleC="none", crossvalI=cross, permI=0)` with
`cross = 7`, or `nrow(des.mat2) - 2` when there are fewer than 7 rows.
`scaleC="none"` because MORE has already scaled. `predI` is left `NA` → autofit.

### 3.1 NIPALS, single response

With one Y column the inner `repeat` exits on its first pass (`ropls:281`), so per
component `h`:

```
w = X'u / (u'u)         where u = y (the single column)
w = w / ||w||
t = X w
c = y't / (t't)
p = X't / (t't)
R2X[h] = ||t p'||² / ||X₀||²
R2Y[h] = ||t c'||² / ||y₀||²
```

then deflate `X ← X - t p'`, `y ← y - t c'`, `rss ← ||y - t c'||²`.

### 3.2 Component selection (`ropls:394-403, 416-446`)

```
ru1Thr = if n > 100 { 0.0 } else { 0.05 }        # orthoI == 0
autMax = min(10, nrow(X), ncol(X))
keep component h iff R2Y[h] >= 0.01 && Q2[h] >= ru1Thr
stop at the first h that fails; that component is discarded
```

`Q2[h] = 1 - PRESS/rss`, with PRESS accumulated over `crossvalI` folds.

**The folds are deterministic**: `split(1:n, rep(1:crossvalI, length = n))`
(`ropls:235`) — sample `i` is in fold `((i-1) mod crossvalI) + 1`. No sampling,
no RNG. Each fold fits **one** component by the same NIPALS step on the in-fold
rows and predicts the held-out rows as `X_out w c'`.

If zero components are significant, `.coreF` returns an object with an empty
`modelDF`; MORE detects that and refits with `predI = 1` forced
(`ResultsPerTargetF.i:112-119`). If that also fails, the target is recorded
`No significant components on PLS`.

### 3.3 Coefficients and VIP

```
R = W (P'W)⁻¹            (R = W when predI == 1)
B = R C'                 -> coefficientMN
ssy[j] = ||t_j c_j'||²
VIP = sqrt( p · Σⱼ (w_j² · ssy[j]) / Σⱼ ssy[j] )      p = number of X columns
```

### 3.4 Jackknife p-values (`p.valuejack`, `auxFunctions.R:652`)

```
k = predI of the main fit
for i in 1..n:
    refit opls(X[-i,], scale(y[-i]), scaleC="none", predI=k, crossvalI=1, permI=0)
    coefficients missing from the refit are padded with 0 and reordered to match
    the main fit's variable order
SE = sqrt( (n-1)/n · Σᵢ (bᵢ - b)² )
p  = 2 · pt(|b / SE|, df = n-1, lower.tail = FALSE)
```

The response is re-scaled *within* each fold (`scale(datospls[-i,1])`), and
`crossvalI=1` disables Q2 in the jackknife fits.

Total fits per target: `1 main + n jackknife + 1 refit on the significant set = n + 2`.
Measured: 22 at n=20, matching exactly.

### 3.5 Significance and the refit

```
sig = { v : VIP[v] > vip } ∩ { v : p[v] < alfa }
```

If `sig` is non-empty the model is refitted on `sig` only, and that refit supplies
the reported `GoodnessOfFit` (`R2Y(cum)`, `Q2(cum)`, `RMSEE`, `NRMSE`, `ncomp`, `sigReg`).
Significant *regulators* are recovered from significant *variables* by
`strsplit(v, ":", fixed=TRUE)` and intersecting with the known regulator IDs —
this is how an interaction term `Group_Treat:TF-1` credits regulator `TF-1`.

**No RNG is reachable on this path.** `set.seed(123)` in `more()` affects only
`varSel="Perm"` and the MLR/elastic-net path. PLS1+Jack is fully deterministic
given the inputs, so the equivalence harness can demand tight numerics here and
must not attribute any edge-set difference to seed variation.

## 4. MLR (`method="MLR"`, `varSel="EN"`) — NOT PORTED, and why the
##    acceptance criterion has to change

`GetMLR` → `ResultsPerTargetF.i.mlr` → `ElasticNet`. It differs from PLS1 in two
ways that matter far more than the change of estimator:

1. it applies `CollinearityFilter1/2` (`correlation = 0.7`), which introduces
   collinearity *groups* with `_R`/`_P`/`_N` representative markers that
   `RegulationPerCondition` resolves through an entirely separate code path
   (`output_analysis.R:92-278`);
2. **it consumes R's RNG heavily.**

### 4.1 The RNG dependence is not incidental

`runMORE.R` never passes `alfaEN`, so `more()`'s default `alfaEN = NULL` reaches
`ElasticNet`, which takes the `is.null(elasticnet)` branch and runs
**eleven `cv.glmnet` fits per target** — `alphas = seq(0, 1, 0.1)` — picking the
alpha whose `cvup` at `lambda.min` is smallest. Every one of those calls draws
its own cross-validation folds from R's Mersenne-Twister, seeded once by
`set.seed(123)` in `more()` and then advanced sequentially across targets.

A different fold assignment moves `lambda.min`, which moves the selected
variable set, which moves the edge set.

### 4.2 Measured, not assumed

Same input, 12 targets x 12 regulators x 20 samples, via `more(method="MLR",
varSel="EN")`:

| comparison | result |
| --- | --- |
| seed 123 run twice | **identical** — R is deterministic *given* a seed |
| seed 123 vs seed 456 | 72 vs 80 edges, symmetric difference **12**, Jaccard **0.854** |

So roughly 15% of the edge set is seed-dependent *inside R itself*.

### 4.3 Consequence for the acceptance criterion

The port brief requires significant edge sets to be **set-equal or hard fail**.
On the PLS1 path that is achievable and achieved — that path reaches no RNG at
all, and the harness measures R's seed spread there as exactly **0 edges**.

On the MLR path it is achievable **only** by reimplementing R's Mersenne-Twister
and `sample.int` fold assignment, glmnet's lambda-path construction, its
coordinate-descent convergence rule at `thres = 1e-5`, and the exact order in
which the RNG stream is advanced across targets. Anything short of that produces
a symmetric difference whose mechanism is "different CV folds" — a real,
identifiable, non-numerical mechanism, but not one that can be tuned away.

The brief's other criterion — *precision/recall within R's own seed-to-seed
spread, measured not assumed* — is the one that can be satisfied, and §4.2 is
the measurement it would be scored against.

**This is a decision for the maintainer, not one to take silently**, because
shipping an MLR that quietly returns a different 15% of the edges is precisely
the "speedup with different biology" the brief calls a failure. Until it is
taken, `--method MLR` exits with a message pointing at `runMORE.R` rather than
doing something different under the same name.

Reference sources for whichever route is chosen are dumped alongside this spec's
notes: `GetMLR`, `ElasticNet`, `ResultsPerTargetF.i.mlr`, `CollinearityFilter1`,
`CollinearityFilter2`, `modelcharac`.

### 4.4 Why the current MLR port under-reports — diagnosed

The elastic net is implemented (`src/elasticnet.rs`) and the harness scores it
at Jaccard 0.22-0.34 against R, with **precision 0.67-0.89 but recall
0.25-0.36**. The port selects too few edges.

The cause is *not* the solver, and not alpha selection. `equivalence/
mlr_alpha_probe.R` drives `cv.glmnet` exactly as `ElasticNet` does and shows R's
`cvup` rule choosing **alpha = 1.0 with 2 non-zero coefficients** — a sparse
lasso solution, the same shape this port produces.

The difference is what reaches the rpc table. On the MLR branch
`GetPairs1targetFAllReg` reports `relevantRegulators`, not
`significantRegulators`, and `ResultsPerTargetF.i.mlr:154-179` builds that set
in two steps:

1. `relevantRegulators <- myvariables` — the variables with non-zero coefficients;
2. **collinearity-group expansion**: for any selected variable that is a group
   *representative* (matched against the `filter` column with `_P`/`_N`/`_R`
   stripped), every original regulator in that group is added and the
   representative itself removed.

So two lasso-selected variables can legitimately become a dozen reported edges.
With correlated regulators — which is the normal case in real omics data, and
what the synthetic sets reproduce — `CollinearityFilter1/2` (`correlation = 0.7`)
collapses them into groups, the lasso picks one representative, and the group
expands back out.

**`CollinearityFilter1/2` is therefore the missing piece, and it is a
prerequisite for MLR equivalence, not an optimisation.** It is unimplemented
here. Until it exists the port cannot reproduce the MLR edge set no matter how
good the elastic net is, because the reported set is a function of the grouping,
not only of the selection.

## 5. Output contract (`runMORE.R:501-602`)

Byte-exact. `<seed>` is `--date_seed`, `<name>` is the sanitised omic name
(spaces → underscores).

| file | format |
| --- | --- |
| `MORE_rpc_<seed>.tab` | `write.table(sep="\t", row.names=FALSE, quote=FALSE, na="")` — header row |
| `MORE_relevant_assoc_<name>_<seed>.tab` | 2 columns `target\tregulator`, **no header**, no quotes |
| `MORE_relevant_pairs_<name>_<seed>.tab` | one column, `TARGET:::REGULATOR`, **no header** |
| `MORE_output_<name>_<seed>.tab` | line 1 `# Gene name\t<sample1>\t…`; then `TARGET:::REGULATOR\t<values>` |

Rules that are easy to get wrong:

- `MORE_output_*` and `MORE_relevant_assoc_*` carry **every** input pair whose
  regulator exists in the regulator matrix — not just the significant ones.
  Only `MORE_relevant_pairs_*` is significance-filtered.
- `NA` is written as the literal `NaN` in `MORE_output_*` (PA Step 1 calls
  `float()` on every value; `float("NA")` raises).
- Every file is created even when empty — an absent file means MORE never ran.
- `rpc` gains an `R2` column merged from `GlobalSummary$GoodnessOfFit`
  (`RsquaredY` for PLS, `Rsquared` for MLR, renamed to `R2`).

## 6. CLI contract

Exactly `runMORE.R`'s `optparse` surface:

```
-t/--target_file   -c/--condition_file   -o/--omic_names
-d/--data_files    -a/--assoc_files      --min_variation
-m/--method        --alpha               --vip
--filter_r2        --output_dir          --date_seed
```

`--omic_names`, `--data_files`, `--assoc_files` are comma-separated and
positionally aligned. `--assoc_files` entries may be the literal `NULL`.
`--min_variation` is one token per omic, or a single token recycled to all;
`NA`/non-numeric means "auto" (10% of maximum observed variability).

Input parsing must reproduce `read_matrix`: try tab and comma, keep whichever
yields **more data columns**, reject a parse with zero data columns, zero rows,
or any non-numeric cell. Sample alignment is a strict name-based intersection —
positional fallback is deliberately absent and must stay absent.
