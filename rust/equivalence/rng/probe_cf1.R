# Dump what CollinearityFilter1 actually receives and produces, from inside a
# live more() run. §4.10 of SPEC.md records that reconstructing its input from
# the generator formula gives a different matrix than the filter really sees,
# so this wraps the real function rather than recomputing outside.
suppressPackageStartupMessages({ library(MORE) })
ns <- asNamespace("MORE")
orig <- get("CollinearityFilter1", ns)
dumped <- FALSE
wrapped <- function(data, reg.table, correlation = 0.8, omic.type, scale, center) {
  if (!dumped) {
    dumped <<- TRUE
    rt <- reg.table; row.names(rt) <- rt[, "regulator"]
    data2 <- scale(data, scale, center)
    myreg <- as.character(rt[which(rt[, "filter"] == "Model"), "regulator"])
    cat("MYREG order:", paste(myreg, collapse = ","), "\n")
    cat("DATA dim:", paste(dim(data), collapse = "x"),
        " cols:", paste(head(colnames(data), 25), collapse = ","), "\n")
    mc <- data.frame(t(combn(myreg, 2)),
                     combn(myreg, 2, function(x) MORE:::correlations(x, data2, rt, omic.type)))
    keep <- mc[abs(mc[, 3]) >= correlation, ]
    cat("EDGES", nrow(keep), "of", nrow(mc), "at |r| >=", correlation, "\n")
    for (i in seq_len(nrow(keep)))
      cat(sprintf("  E %s %s %.10f\n", keep[i,1], keep[i,2], keep[i,3]))
  }
  res <- orig(data, reg.table, correlation, omic.type, scale, center)
  res
}
assignInNamespace("CollinearityFilter1", wrapped, ns = "MORE")

NS<-20; NR<-20; NT<-12
S<-paste0("S",1:NS)
reg <- t(sapply(1:NR, function(j) ((0:(NS-1))*7 + (j-1)*13) %% 23 / 23 - 0.5))
rownames(reg)<-paste0("R",1:NR); colnames(reg)<-S
tgt <- t(sapply(1:NT, function(g) { a <- 3*reg[((g-1)%%NR)+1,] + 1.5*reg[((g)%%NR)+1,]
  a + 0.05*(((g*5+0:(NS-1))%%11)/11 - 0.5)}))
rownames(tgt)<-paste0("G",1:NT); colnames(tgt)<-S
cond<-data.frame(Ctrl=c(rep(1,NS/2),rep(0,NS/2)),Treat=c(rep(0,NS/2),rep(1,NS/2))); rownames(cond)<-S
assoc<-do.call(rbind,lapply(1:NT,function(g) data.frame(target=paste0("G",g),regulator=paste0("R",1:NR))))
res<-more(targetData=tgt, regulatoryData=list(TF=as.data.frame(reg)), associations=list(TF=assoc),
          condition=cond, method="MLR", varSel="EN", minVariation=c(TF=0), alfa=0.05, seed=123)
r <- res$ResultsPerTargetF[["G1"]]
cat("G1 filter values:\n"); print(table(r$allRegulators[,"filter"]))
