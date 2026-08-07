suppressPackageStartupMessages({library(glmnet)})
set.seed(123)
n <- 20; p <- 26
X <- matrix(0, n, p)
for (i in 1:n) for (j in 1:p) X[i,j] <- ((i-1)*7 + (j-1)*13) %% 23 / 23 - 0.5
colnames(X) <- paste0("V", 1:p)
y <- 3*X[,1] + 0.05*(((1:n)*5) %% 11 / 11 - 0.5)
alphas <- seq(0,1,0.1); cvupmin <- .Machine$double.xmax; best <- NA; bestnz <- NA
for (a in alphas) {
  cv <- cv.glmnet(X, y, nfolds=n, alpha=a, standardize=FALSE, thres=1e-5, family=gaussian(), grouped=FALSE)
  cu <- cv$cvup[which(cv$lambda == cv$lambda.min)]
  nz <- sum(coef(cv, s=cv$lambda.min)[-1,1] != 0)
  cat(sprintf("alpha=%.1f  lambda.min=%.6g  cvup=%.6g  nonzero=%d\n", a, cv$lambda.min, cu, nz))
  if (cu < cvupmin) { cvupmin <- cu; best <- a; bestnz <- nz }
}
cat(sprintf("WINNER alpha=%.1f  nonzero=%d of %d\n", best, bestnz, p))
