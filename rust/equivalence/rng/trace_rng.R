# Ground truth for the MLR RNG stream.
#
# Every draw MORE and glmnet make on the MLR path goes through base::sample.
# base's binding is locked, so we unlock it and install a pass-through shim that
# records (call site, input vector, size, result) in order. The shim must never
# call `sample` itself or it recurses -- `orig` is captured first.
#
# Output: rng_trace.tsv, one row per draw, in stream order. This is the file the
# Rust port has to reproduce exactly.
suppressPackageStartupMessages({ library(MORE) })

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "."
dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

orig <- base::sample
.n <- 0L
.rows <- list()
shim <- function(x, size, replace = FALSE, prob = NULL) {
  res <- if (missing(size)) orig(x, replace = replace, prob = prob)
         else orig(x, size = size, replace = replace, prob = prob)
  .n <<- .n + 1L
  # Caller identity: the innermost frame that is not this shim.
  site <- tryCatch({
    cl <- sys.call(-1)
    if (is.null(cl)) "?" else substr(paste(deparse(cl), collapse=" "), 1, 60)
  }, error = function(e) "?")
  n_in <- if (length(x) == 1L && is.numeric(x) && is.finite(x) && x >= 1) as.integer(x) else length(x)
  .rows[[length(.rows) + 1L]] <<- data.frame(
    seq = .n,
    n = n_in,
    size = if (missing(size)) n_in else as.integer(size),
    input = paste(as.character(x), collapse = ","),
    result = paste(as.character(res), collapse = ","),
    site = site,
    stringsAsFactors = FALSE)
  res
}
unlockBinding("sample", baseenv())
assign("sample", shim, envir = baseenv())

# --- dataset: the mlr_internals_probe shape, fully deterministic -------------
NS <- 20; NR <- 20; NT <- 12
S <- paste0("S", 1:NS)
reg <- t(sapply(1:NR, function(j) ((0:(NS-1))*7 + (j-1)*13) %% 23 / 23 - 0.5))
rownames(reg) <- paste0("R", 1:NR); colnames(reg) <- S
tgt <- t(sapply(1:NT, function(g) {
  a <- 3*reg[((g-1)%%NR)+1, ] + 1.5*reg[((g)%%NR)+1, ]
  a + 0.05*(((g*5+0:(NS-1))%%11)/11 - 0.5) }))
rownames(tgt) <- paste0("G", 1:NT); colnames(tgt) <- S
cond <- data.frame(Ctrl = c(rep(1, NS/2), rep(0, NS/2)), Treat = c(rep(0, NS/2), rep(1, NS/2)))
rownames(cond) <- S
assoc <- do.call(rbind, lapply(1:NT, function(g)
  data.frame(target = paste0("G", g), regulator = paste0("R", 1:NR))))

# Dump the same input the port will read.
write.table(data.frame(ID = rownames(tgt), tgt, check.names = FALSE), file.path(outdir, "targets.tab"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(ID = rownames(reg), reg, check.names = FALSE), file.path(outdir, "tf.tab"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(data.frame(ID = rownames(cond), cond, check.names = FALSE), file.path(outdir, "cond.tab"),
            sep = "\t", quote = FALSE, row.names = FALSE)
write.table(assoc, file.path(outdir, "assoc.tab"), sep = "\t", quote = FALSE, row.names = FALSE)

res <- more(targetData = tgt, regulatoryData = list(TF = as.data.frame(reg)),
            associations = list(TF = assoc), condition = cond,
            method = "MLR", varSel = "EN", minVariation = c(TF = 0), alfa = 0.05, seed = 123)

trace_df <- do.call(rbind, .rows)
write.table(trace_df, file.path(outdir, "rng_trace.tsv"), sep = "\t", quote = FALSE, row.names = FALSE)
cat("draws:", nrow(trace_df), "\n")
print(table(sub("\\(.*", "", trace_df$site)))
