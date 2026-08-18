# cv.glmnet's own lambda/cvm/cvsd curve at one alpha, for diffing against
# more-rs's cv_curve on the identical design matrix.
suppressPackageStartupMessages(library(glmnet))
a <- commandArgs(trailingOnly = TRUE)
m <- read.delim(a[1], check.names = FALSE)
alpha <- as.numeric(a[2])
nf <- ifelse(nrow(m) < 50, nrow(m), ifelse(nrow(m) < 100, 5, ifelse(nrow(m) < 200, 7, 10)))
cv <- cv.glmnet(x = as.matrix(m[, -1]), y = m[, 1], nfolds = nf, alpha = alpha,
                standardize = FALSE, thres = 1e-5, family = "gaussian", grouped = FALSE)
nz <- sapply(seq_along(cv$lambda), function(i) sum(coef(cv$glmnet.fit)[-1, i] != 0))
for (i in seq_along(cv$lambda))
  cat(sprintf("   %3d lambda=%.10e cvm=%.10e cvsd=%.10e nz=%d\n",
              i, cv$lambda[i], cv$cvm[i], cv$cvsd[i], nz[i]))
