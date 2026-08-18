# Records every base::sample draw, appended as it happens so no finaliser is
# needed and a mid-run abort still leaves a usable prefix. Prepended to a copy
# of runMORE.R so the trace covers the real product seam: same argv, same file
# parsing, same code path MOREServlet invokes.
local({
  orig <- base::sample
  path <- Sys.getenv("MORE_R_RNG_TRACE", "r_trace.tsv")
  cat("seq\tn\tsize\tinput\tresult\n", file = path)
  n <- 0L
  shim <- function(x, size, replace = FALSE, prob = NULL) {
    res <- if (missing(size)) orig(x, replace = replace, prob = prob)
           else orig(x, size = size, replace = replace, prob = prob)
    n <<- n + 1L
    n_in <- if (length(x) == 1L && is.numeric(x) && is.finite(x) && x >= 1) as.integer(x) else length(x)
    cat(sprintf("%d\t%d\t%d\t%s\t%s\n", n, n_in,
                if (missing(size)) n_in else as.integer(size),
                paste(as.character(x), collapse = ","),
                paste(as.character(res), collapse = ",")),
        file = path, append = TRUE)
    res
  }
  unlockBinding("sample", baseenv())
  assign("sample", shim, envir = baseenv())
})

# Optional: dump every design matrix ElasticNet receives, and the alpha it
# picked, so the port can be handed the identical matrix and the two CV
# decisions compared directly rather than inferred from the output files.
if (nzchar(Sys.getenv("MORE_R_DESIGN_DIR"))) local({
  dir <- Sys.getenv("MORE_R_DESIGN_DIR")
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  ns <- asNamespace("MORE")
  orig <- get("ElasticNet", ns)
  i <- 0L
  wrap <- function(family2, des.mat2, epsilon, elasticnet) {
    i <<- i + 1L
    write.table(des.mat2, file.path(dir, sprintf("des_%02d.tsv", i)),
                sep = "\t", quote = FALSE, row.names = FALSE)
    # MORE_R_EPSILON forces the convergence tolerance glmnet is given. MORE's
    # own default is 1e-5, which leaves glmnet measurably short of the optimum;
    # overriding it is the only way to compare R against another implementation
    # with both actually converged.
    eps <- Sys.getenv("MORE_R_EPSILON")
    if (nzchar(eps)) epsilon <- as.numeric(eps)
    res <- orig(family2, des.mat2, epsilon, elasticnet)
    cat(sprintf("RCHOSE %02d ncol=%d alpha=%s R2=%s\n", i, ncol(des.mat2),
                as.character(res$elasticnet),
                if (!is.null(res$m)) as.character(round(res$m$R.squared, 6)) else "NA"),
        file = file.path(dir, "chosen.txt"), append = TRUE)
    res
  }
  assignInNamespace("ElasticNet", wrap, ns = "MORE")
})
