## Dump every design matrix MORE hands to ElasticNet, for the *same* input
## files the harness feeds the binary.
##
## This traces MORE:::ElasticNet and then sources runMORE.R unmodified, so the
## capture goes through the real entry point rather than a reconstruction of
## it -- the R sources stay the oracle and are never edited.
##
## Usage: MORE_RUNMORE=<runMORE.R> MORE_DESIGN_OUT=<dir> \
##            Rscript design_probe.R <runMORE args...>
##
## runMORE.R's own optparse call reads commandArgs() from inside the optparse
## namespace, so the args cannot be shadowed from here -- they have to be the
## real ones. Hence the script path arrives by environment variable.
##
## Matrices are written as en_<i>.tsv with the response in column 1; the
## diffing side identifies which target each one belongs to by matching that
## response column, so call order never has to be assumed.

suppressPackageStartupMessages({library(MORE); library(optparse)})

outdir <- Sys.getenv("MORE_DESIGN_OUT", unset = "design_dump")
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

runmore <- Sys.getenv("MORE_RUNMORE")
if (!nzchar(runmore) || !file.exists(runmore))
  stop("set MORE_RUNMORE to the path of runMORE.R")

counter <- new.env(parent = emptyenv())
counter$i <- 0L
assign(".more_design_counter", counter, envir = globalenv())
assign(".more_design_out", outdir, envir = globalenv())

## The collinearity filter's own answer: SummaryPerTargetF's `filter` column
## carries `<omic>_mc<i>_R` on the representative and `_P`/`_N` on the other
## members, so one dump per target fixes both membership and the sign.
trace(MORE:::CollinearityFilter1, exit = quote({
  cnt <- get(".more_design_counter", envir = globalenv())
  rv <- returnValue()
  if (!is.null(rv)) {
    write.table(rv$SummaryPerTargetF,
                file.path(get(".more_design_out", envir = globalenv()),
                          sprintf("cf_%03d.tsv", cnt$i + 1L)),
                sep = "\t", quote = FALSE, row.names = FALSE)
  }
}), print = FALSE)

trace(MORE:::ElasticNet, tracer = quote({
  cnt <- get(".more_design_counter", envir = globalenv())
  cnt$i <- cnt$i + 1L
  write.table(des.mat2,
              file.path(get(".more_design_out", envir = globalenv()),
                        sprintf("en_%03d.tsv", cnt$i)),
              sep = "\t", quote = FALSE, row.names = FALSE)
}), print = FALSE)

source(runmore)
cat(sprintf("captured %d design matrices into %s\n",
            get(".more_design_counter", envir = globalenv())$i, outdir))
