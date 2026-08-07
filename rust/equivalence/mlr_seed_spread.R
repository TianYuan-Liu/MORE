suppressPackageStartupMessages({library(MORE)})
set.seed(1)
NS <- 20; NR <- 12; NT <- 12
S <- paste0("S", 1:NS)
reg <- t(sapply(1:NR, function(j) ((0:(NS-1))*7 + j*13) %% 23 / 23 - 0.5))
rownames(reg) <- paste0("R", 1:NR); colnames(reg) <- S
tgt <- t(sapply(1:NT, function(g) 3*reg[((g-1) %% NR)+1, ] + 0.05*(((g*5+0:(NS-1)) %% 11)/11 - 0.5)))
rownames(tgt) <- paste0("G", 1:NT); colnames(tgt) <- S
cond <- data.frame(Ctrl=c(rep(1,NS/2),rep(0,NS/2)), Treat=c(rep(0,NS/2),rep(1,NS/2)))
rownames(cond) <- S
assoc <- do.call(rbind, lapply(1:NT, function(g) data.frame(target=paste0("G",g), regulator=paste0("R",1:NR))))

edges <- function(seed) {
  res <- more(targetData=tgt, regulatoryData=list(TF=as.data.frame(reg)),
              associations=list(TF=assoc), condition=cond,
              method="MLR", varSel="EN", minVariation=c(TF=0), alfa=0.05, seed=seed)
  rpc <- RegulationPerCondition(res)
  unique(paste(rpc$targetF, rpc$regulator, sep=":::"))
}
a <- edges(123); b <- edges(456); c3 <- edges(123)
cat("seed123 edges:", length(a), "\n")
cat("seed456 edges:", length(b), "\n")
cat("seed123 repeat identical:", identical(sort(a), sort(c3)), "\n")
cat("symmetric difference 123 vs 456:", length(union(setdiff(a,b), setdiff(b,a))), "\n")
cat("jaccard:", round(length(intersect(a,b))/length(union(a,b)), 4), "\n")
