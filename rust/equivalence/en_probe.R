## Capture the exact design matrices MORE hands to ElasticNet, then report
## what cv.glmnet decides for each of the eleven alphas.
##
## The Rust side reads the very same TSVs (MORE_RS_EN_PROBE=<file> more-rs)
## and prints the same columns, so the two can be diffed row by row. This is
## the instrument called for by SPEC.md 4.11.3: the mlr-denser divergence is
## in cross-validated (alpha, lambda) selection, and guessing at it twice
## already cost more than measuring it once.
##
## Usage: Rscript en_probe.R <outdir>

suppressPackageStartupMessages({library(MORE); library(glmnet)})

args <- commandArgs(trailingOnly = TRUE)
outdir <- if (length(args) >= 1) args[1] else "fixtures/en"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

## Same generator as mlr_internals_probe.R / the mlr-denser parameter set.
NS <- 20; NR <- 20; NT <- 12
S <- paste0("S", 1:NS)
reg <- t(sapply(1:NR, function(j) ((0:(NS-1))*7 + (j-1)*13) %% 23 / 23 - 0.5))
rownames(reg) <- paste0("R", 1:NR); colnames(reg) <- S
tgt <- t(sapply(1:NT, function(g) {
  a <- 3*reg[((g-1) %% NR)+1, ] + 1.5*reg[((g) %% NR)+1, ]
  a + 0.05*(((g*5+0:(NS-1)) %% 11)/11 - 0.5)}))
rownames(tgt) <- paste0("G", 1:NT); colnames(tgt) <- S
cond <- data.frame(Ctrl = c(rep(1, NS/2), rep(0, NS/2)),
                   Treat = c(rep(0, NS/2), rep(1, NS/2)))
rownames(cond) <- S
assoc <- do.call(rbind, lapply(1:NT, function(g)
  data.frame(target = paste0("G", g), regulator = paste0("R", 1:NR))))

## Capture every des.mat2 that reaches ElasticNet, in call order.
captured <- new.env(parent = emptyenv())
captured$mats <- list()
trace(MORE:::ElasticNet, tracer = quote({
  captured <- get("captured", envir = globalenv())
  captured$mats[[length(captured$mats) + 1L]] <- des.mat2
}), print = FALSE)

invisible(capture.output(
  res <- more(targetData = tgt, regulatoryData = list(TF = as.data.frame(reg)),
              associations = list(TF = assoc), condition = cond,
              method = "MLR", varSel = "EN", minVariation = c(TF = 0),
              alfa = 0.05, seed = 123)))
untrace(MORE:::ElasticNet)

mats <- captured$mats
cat(sprintf("captured %d design matrices\n", length(mats)))

alphas <- seq(0, 1, 0.1)
for (i in seq_along(mats)) {
  m <- mats[[i]]
  f <- file.path(outdir, sprintf("des_%02d.tsv", i))
  write.table(m, f, sep = "\t", quote = FALSE, row.names = FALSE)
  cat(sprintf("== matrix %d  %d x %d  response=%s  -> %s\n",
              i, nrow(m), ncol(m) - 1L, colnames(m)[1], f))
  ## Fold count is MORE's own rule; with n < 50 this is leave-one-out, which
  ## makes cv.glmnet's sample() a relabelling and the whole branch deterministic.
  mynfolds <- ifelse(nrow(m) < 50, nrow(m),
              ifelse(nrow(m) < 100, 5, ifelse(nrow(m) < 200, 7, 10)))
  if (ncol(m) <= 2) { cat("   (too few columns for EN)\n"); next }
  cvupmin <- .Machine$double.xmax; best <- NA
  for (a in alphas) {
    cv <- cv.glmnet(x = as.matrix(m[, -1]), y = m[, 1], nfolds = mynfolds,
                    alpha = a, standardize = FALSE, thres = 1e-5,
                    family = "gaussian", grouped = FALSE)
    k <- which(cv$lambda == cv$lambda.min)
    nz <- sum(coef(cv, s = cv$lambda.min)[-1, 1] != 0)
    cat(sprintf("   a=%.1f nlam=%3d lmax=%.6g lmin_path=%.6g lambda.min=%.6g cvm=%.6g cvup=%.6g nz=%d\n",
                a, length(cv$lambda), cv$lambda[1], cv$lambda[length(cv$lambda)],
                cv$lambda.min, cv$cvm[k], cv$cvup[k], nz))
    if (cv$cvup[k] < cvupmin) { cvupmin <- cv$cvup[k]; best <- a }
  }
  cat(sprintf("   WINNER a=%.1f cvup=%.6g\n", best, cvupmin))
}
