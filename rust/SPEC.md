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

### 4.5 `CollinearityFilter1` — the algorithm to implement

`GetMLR` passes `col.filter = "cor"`, so `CollinearityFilter1` is the one on the
PaintOmics path (`ResultsPerTargetF.i.mlr:76`); `CollinearityFilter2` (`"pcor"`,
partial correlations) is unreachable from `runMORE.R`. It runs only when the
target has more than one Model regulator.

1. scale the regulator matrix (`scale(data, scale, center)`);
2. for every pair of `filter == "Model"` regulators compute a correlation whose
   *kind* depends on the two omic types (`correlations`):
   - numeric/numeric → Pearson `cor`
   - numeric/binary → `ltm::biserial.cor`
   - binary/binary → `psych::phi` on the contingency table
3. keep pairs with `|r| >= correlation` (0.7 from `more()`'s default);
4. build an undirected graph on those pairs and take its connected components;
5. **collapse a component only if it is a complete clique** —
   `nedges == csize*(csize-1)/2`. A merely connected, non-clique component is
   left alone;
6. pick one member as the representative, drop the other columns, and rename the
   survivor `<omic>_mc<i>_R`. Append a duplicate row to the regulator table under
   that name, set the representative's own `filter` to it, and set each dropped
   member's `filter` to `<omic>_mc<i>_P` or `_N` according to the sign of its
   correlation with the representative.

Then, after elastic-net selection, `ResultsPerTargetF.i.mlr:170-179` expands any
selected `*_mc<i>_R` variable back to **every** regulator whose `filter` names
that group, and removes the representative itself. That expansion is the whole
recall gap in §4.4.

#### 4.5.1 A third RNG site

Step 6 uses `keep = sample(correlacionados, 1)` — the representative is drawn at
**random**, from the same stream `set.seed(123)` seeds. This is a third RNG
dependence on the MLR path, after `cv.glmnet`'s folds and its alpha loop.

It is the most benign of the three for edge-set purposes: the *membership* of a
group is deterministic (a clique in the correlation graph), and the expansion in
`relevantRegulators` returns every member regardless of which one was drawn. So
the reported edge set should be stable under this draw even though the retained
column, its name, and therefore the coefficient attribution are not.

A port may therefore choose the representative deterministically — first member
in canonical order — and still expect edge-set agreement, while the per-condition
coefficients for a collapsed group will differ because they are attached to a
different member. That difference has a mechanism, and it is this one; it should
be reported as such rather than absorbed into a tolerance.

### 4.6 Instrumented state — where the port still diverges

`equivalence/mlr_internals_probe.R` dumps R's per-target `relevantRegulators`,
coefficient rownames and `allRegulators$filter` for a 12 x 20 x 20 job. On input
where **this port detects zero cliques**, R produces **six groups**:

```
filter values: Model=8
               TF_mc1_1_P=3  TF_mc1_1_R=1
               TF_mc1_2_P=2  TF_mc1_2_R=1
               TF_mc1_3_P=1  TF_mc1_3_R=1
               TF_mc1_4_P=2  TF_mc1_4_R=1
               TF_mc1_5_P=2  TF_mc1_5_R=1
               TF_mc1_6_P=2  TF_mc1_6_R=1

G3  coefficients: (Intercept), Group_1_0, R6, R16,
                  TF_mc1_1_R, TF_mc1_2_R, TF_mc1_3_R, TF_mc1_4_R, ...
    relevantRegulators: all 20
```

Three concrete facts to work from, none of them guesses:

1. **The clique detection is wrong.** Six groups covering 12 of 20 regulators
   versus zero found. Either the correlation is being computed over the wrong
   vectors, or the complete-clique test rejects components R accepts. Note the
   group sizes here are 2-4, so they are small cliques, not one large component.
2. **The group naming is `<omic>_mc1_<i>_R`, not `<omic>_mc<i>_R`.** The port
   emits the latter. This matters because `ResultsPerTargetF.i.mlr` matches
   selected variables against the `filter` column by exactly this string.
3. **Grouping precedes interaction construction.** R's coefficient rownames
   include `Group_1_0:TF_mc1_1_R` — interactions are built on the *representative*
   column, after collapsing. The port collapses before `design::build` too, so
   this ordering is right, but it must stay that way.

The port currently reaches Jaccard 0.4632 (mlr-small) and 0.3434 (mlr-denser)
against R's own 0.854 seed band. Fixing (1) is the next step, and (2) must land
with it or the expansion will not match even once the cliques are right.

### 4.7 Both sides dumped — the observation, without a theory attached

`MORE_RS_DEBUG_MLR=1` makes the binary emit the port's counterpart of what
`equivalence/mlr_internals_probe.R` emits for R. On identical input
(12 targets x 20 regulators x 20 samples, `minVariation = 0`):

| | R | port |
| --- | --- | --- |
| Model regulators per target | 20 | 20 |
| collinearity groups | **6**, sizes 2-4, covering 12 regulators | **0** |
| design columns | — | 62 (20 regulators + 40 interactions + 2 design) |

Independently measured on the same regulator matrix: **21 pairs at
`|r| >= 0.7`, max `|r| = 1.0`**. So the edges are present and the port's
correlations are not the problem; its *component handling* is. A single pass
that rejects any component failing `nedges == csize*(csize-1)/2` finds nothing,
where R arrives at six small cliques.

The obvious reconciliation was that R peels large components across repeated
applications. **That hypothesis is FALSIFIED.** `MORE_MLR.R:606-611` calls
`CollinearityFilter1` exactly once per target, guarded only by
`ncol(res$RegulatorMatrix) > 1` — there is no loop and no re-entry. R produces
all six cliques in a single pass.

That relocates the defect to the graph itself. For R to obtain six *complete*
components in one pass, its `|r| >= 0.7` graph must be close to six disjoint
cliques (sizes 2-4 imply roughly 19 edges, against the 21 measured). The port
sees 21 edges over the same 20 nodes and yields **zero** complete components,
which is what happens when a handful of extra edges bridge otherwise-disjoint
cliques into one large non-complete blob that the completeness test then rejects
wholesale.

### 4.8 The clique code is exonerated; the INPUT matrix is the suspect

The edge-set diff was done, in R, using R's own `igraph` calls on the raw
regulator matrix from the probe:

```
edges: 21    components: 1    sizes: 20    complete components: 0 of 1
```

So R's *algorithm*, applied to the raw regulator matrix, collapses **nothing** —
byte-identical in outcome to this port, which also finds zero groups on that
matrix. The port's correlation, threshold, component and completeness logic all
agree with R's on that input.

But R's actual run over the same job produced **six** groups
(`TF_mc1_1_R` … `TF_mc1_6_R`). Both statements are measured. The only way both
hold is that `CollinearityFilter1` is **not** receiving the raw regulator
matrix.

That narrows the remaining work sharply, and away from `collinearity.rs`:

* `res$RegulatorMatrix` comes from `RemovedRegulators`, which builds it as
  `t(data.omics[[ov]][regmodel, , drop = FALSE])` per omic and `cbind`s the
  blocks. Confirm the column set and orientation the port reproduces.
* `CollinearityFilter1` correlates `scale(data, scale, center)` — Pearson is
  scale-invariant, so this cannot be the difference, and is ruled out.
* The correlation *kind* is chosen per pair by `omicType`. **Checked and
  falsified**: `MORE:::isBin` returns **0** for this omic (20 distinct values in
  both the first row and the first column), so R uses Pearson, same as the port.

### 4.9 What survives: the filter sees the interaction-expanded matrix

Every other candidate is now eliminated by measurement:

| candidate | verdict |
| --- | --- |
| elastic-net solver / alpha selection | ruled out — R picks alpha=1.0, 2 non-zero, same shape as the port |
| group expansion missing | implemented; real but partial (0.2247 → 0.4632) |
| iterative peeling of components | falsified — `MORE_MLR.R:606-611` calls the filter once, no loop |
| correlation values / threshold / clique logic | ruled out — R's own igraph run on the raw matrix gives 0 complete components, matching the port |
| binary omic → phi/biserial correlation | falsified — `isBin` returns 0 |

One explanation remains consistent with all of it. R's groups are named
`TF_mc1_1_R … TF_mc1_6_R`: **six** groups of size **2-4**. The raw 20-regulator
matrix yields a single 20-node component with no complete subcomponent. But the
interaction-expanded design has 62 columns — 20 regulators, 40
`Group_*:regulator` interactions, 2 design columns — and a main effect `R1` is
near-perfectly correlated with its own `Group_A:R1` and `Group_B:R1` and with
little else. That structure produces precisely small disjoint cliques of size
2-4, in the observed count.

**This too is falsified.** `MORE_MLR.R` orders them the other way:

```
609  res = CollinearityFilter1(data = res$RegulatorMatrix, ...)
633  des.mat2EN = RegulatorsInteractions(interactions, reguValues = res$RegulatorMatrix, ...)
```

The filter runs *before* interactions are built, so it does receive the raw
regulator matrix. The port's ordering was right all along.

### 4.11 RESOLVED: non-complete components are star-peeled, not discarded

Tracing `CollinearityFilter1` inside a live run showed it receives exactly the
raw 20 x 20 regulator matrix (`FILTER-IN dim: 20 20`, columns `R1..R20`), which
is the matrix whose igraph analysis reports zero complete components. The
contradiction in §4.10 was therefore *inside the filter*, in a branch that had
not been read.

`CollinearityFilter1` handles a non-complete component in an `else` branch that
**iteratively peels stars** rather than discarding it:

```
while not every node is isolated:
    repre  = highest-degree node
             ties broken by max sum |r| over its edges, then by sample()
    absorb every neighbour of repre into one group
    name it <omic>_mc<i>_<j>_R
    remove them; j = j + 1
```

That is the source of the `TF_mc1_1_R … TF_mc1_6_R` naming — component `i = 1`,
peels `j = 1..6` — and it turns a single 20-node component with 21 edges into
six groups. The port previously rejected such components outright, which is why
it found zero.

Implemented in `collinearity.rs`. The port now produces six groups with R's
names and the same group-size multiset on the probe job, and the harness moved:

| set | before | after |
| --- | --- | --- |
| mlr-small | 0.4632 | **0.6304** |
| mlr-denser | 0.3434 | **0.5931** |

against R's own 0.854 seed band. Still failing, so MLR remains unshippable, but
the mechanism was real and is now in place.

The residual is whole targets differing — R models a target the port does not
and vice versa — which is the expected signature of the `sample()` tie-breaks
inside the peel loop changing group composition, and therefore which regulators
enter the model at all. That is the fourth RNG site on this path and the first
one that can move the edge set, so it is the next thing to characterise: count
how often the degree/correlation tie-break is actually hit before assuming it
explains the gap.

### 4.10 An unresolved contradiction — RESOLVED, see §4.11

Two measurements now conflict, and both were taken directly:

* R's own `igraph` pipeline, on the raw 20-regulator matrix, reports
  **1 component, 21 edges, 0 complete components** — nothing to collapse;
* R's actual `more(method="MLR")` run over a job built from that same matrix
  reports **6 groups**, `TF_mc1_1_R … TF_mc1_6_R`, sizes 2-4.

Both cannot be true of the same input, so the inputs must differ. The
standalone igraph check reconstructed the regulator matrix from the generator
formula; the `more()` run passed it through `GetMLR`'s own pre-filters. The
difference is therefore somewhere in what `GetMLR` does to `regulatoryData`
before `ResultsPerTargetF.i.mlr` sees it — the `Inf` filter, the `percNA`
filter, `LowVariationRegu`, the sample-level NA filter, or the ID mangling —
any of which can change the surviving column set and hence the graph.

**Do not propose another mechanism.** Reproduce the contradiction first: inside
a live `more(method="MLR")` run, dump `dim(res$RegulatorMatrix)`,
`colnames(res$RegulatorMatrix)` and the resulting `mycor` for a single target,
and compare that matrix against the standalone reconstruction. Whichever of the
two the port matches tells you which side is wrong.

Five mechanisms have now been proposed on this path and four were falsified by
measurement — alpha selection, iterative peeling, binary correlation kind, and
interaction ordering. Only group expansion was real, and it was found by
instrumenting rather than by reasoning. The pattern is the finding: on this
code path, reading the source produces plausible wrong answers at a high rate,
and only dumping live intermediates has produced correct ones.

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
