#!/usr/bin/env Rscript
#
# Generate a golden fixture for the Rust PLS1 + jackknife port straight from the
# reference stack: ropls::opls called exactly as MORE calls it, and MORE's own
# p.valuejack.
#
# The design matrix is built from integer arithmetic, not rnorm(), so the same
# bits are reproducible in Rust without transcribing the matrix by hand and
# without depending on either language's RNG. The already-scaled X and y are
# written to the fixture at full precision so the Rust test exercises the PLS
# algorithm alone -- any difference in how the two languages sum a mean cannot
# leak into the comparison.
#
#   Rscript equivalence/pls_oracle.R
#
# writes equivalence/fixtures/pls_oracle.tsv

suppressPackageStartupMessages({
  library(ropls)
  library(MORE)
})

args <- commandArgs(trailingOnly = TRUE)
outDir <- if (length(args) > 0) args[1] else "equivalence/fixtures"
dir.create(outDir, recursive = TRUE, showWarnings = FALSE)

N <- 20L   # samples -- the size PaintOmics jobs actually run at
P <- 6L    # regulator columns

# Deterministic, exactly representable in both languages: an integer modulus
# followed by one IEEE division.
X <- matrix(0, nrow = N, ncol = P)
for (i in seq_len(N)) {
  for (j in seq_len(P)) {
    X[i, j] <- ((i - 1L) * 7L + (j - 1L) * 13L) %% 23L / 23 - 0.5
  }
}
y <- numeric(N)
for (i in seq_len(N)) {
  y[i] <- 0.8 * X[i, 1] - 0.5 * X[i, 3] + 0.15 * (((i - 1L) * 5L) %% 11L / 11 - 0.5)
}

colnames(X) <- paste0("R", seq_len(P))

# MORE scales before calling opls and then passes scaleC = "none".
Xs <- scale(X, center = TRUE, scale = TRUE)
ys <- scale(y, center = TRUE, scale = TRUE)

cross <- if (nrow(Xs) < 7) nrow(Xs) - 2 else 7

myPLS <- suppressWarnings(ropls::opls(
  Xs, ys,
  info.txtC = "none", fig.pdfC = "none", scaleC = "none",
  crossvalI = cross, permI = 0))

stopifnot(length(myPLS@modelDF) > 0)

# p.valuejack wants the response in column 1, as ResultsPerTargetF.i assembles it.
datospls <- data.frame(response = as.numeric(ys), Xs, check.names = FALSE)
pval <- MORE:::p.valuejack(myPLS, datospls, 0.05)

g <- function(v) paste(sprintf("%.17g", as.numeric(v)), collapse = "\t")

lines <- c(
  paste0("# ropls ", as.character(packageVersion("ropls")),
         "  MORE ", as.character(packageVersion("MORE")),
         "  R ", paste0(R.version$major, ".", R.version$minor),
         "  ", R.version$platform),
  paste0("n\t", N),
  paste0("p\t", P),
  paste0("crossval\t", cross),
  paste0("x\t", g(as.vector(Xs))),          # column-major, as R stores it
  paste0("y\t", g(as.numeric(ys))),
  paste0("ncomp\t", myPLS@summaryDF[, "pre"]),
  paste0("coefficients\t", g(myPLS@coefficientMN[, 1])),
  paste0("vip\t", g(myPLS@vipVn)),
  paste0("r2y_cum\t", g(myPLS@modelDF[, "R2Y(cum)"][myPLS@summaryDF[, "pre"]])),
  paste0("q2_cum\t", g(myPLS@modelDF[, "Q2(cum)"][myPLS@summaryDF[, "pre"]])),
  paste0("rmsee\t", g(myPLS@summaryDF[, "RMSEE"])),
  paste0("fitted\t", g(myPLS@suppLs$yPreMN[, 1])),
  paste0("pvalue\t", g(pval[, 1]))
)

out <- file.path(outDir, "pls_oracle.tsv")
writeLines(lines, out)
cat("wrote", out, "\n")
cat("ncomp =", myPLS@summaryDF[, "pre"], " R2Y(cum) =",
    myPLS@modelDF[, "R2Y(cum)"][myPLS@summaryDF[, "pre"]], "\n")
