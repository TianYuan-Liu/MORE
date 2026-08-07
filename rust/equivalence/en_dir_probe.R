## Report what cv.glmnet decides, for design matrices already on disk.
##
## Unlike en_probe.R this does not generate anything: it reads the matrices
## captured by design_probe.R from a real runMORE.R run, so the port and R are
## compared on byte-identical input and any remaining difference is the
## cross-validation itself.
##
## Usage: Rscript en_dir_probe.R <dir-of-tsv>

suppressPackageStartupMessages(library(glmnet))
dir <- commandArgs(trailingOnly = TRUE)[1]
files <- sort(list.files(dir, pattern = "\\.tsv$", full.names = TRUE))
alphas <- seq(0, 1, 0.1)

for (i in seq_along(files)) {
  m <- read.delim(files[i], check.names = FALSE)
  cat(sprintf("== matrix %d\n", i))
  ## MORE's own fold rule; under 50 samples this is leave-one-out, which makes
  ## cv.glmnet's sample() a relabelling and the branch fully deterministic.
  nf <- ifelse(nrow(m) < 50, nrow(m),
        ifelse(nrow(m) < 100, 5, ifelse(nrow(m) < 200, 7, 10)))
  cvupmin <- .Machine$double.xmax; best <- NA
  for (a in alphas) {
    cv <- cv.glmnet(x = as.matrix(m[, -1]), y = m[, 1], nfolds = nf, alpha = a,
                    standardize = FALSE, thres = 1e-5, family = "gaussian",
                    grouped = FALSE)
    k <- which(cv$lambda == cv$lambda.min)
    nz <- sum(coef(cv, s = cv$lambda.min)[-1, 1] != 0)
    cat(sprintf("   a=%.1f nlam=%3d lmax=%.6g lmin_path=%.6g lambda.min=%.6g cvm=%.6g cvup=%.6g nz=%d\n",
                a, length(cv$lambda), cv$lambda[1], cv$lambda[length(cv$lambda)],
                cv$lambda.min, cv$cvm[k], cv$cvup[k], nz))
    if (cv$cvup[k] < cvupmin) { cvupmin <- cv$cvup[k]; best <- a }
  }
  cat(sprintf("   WINNER a=%.1f cvup=%.6g\n", best, cvupmin))
}
