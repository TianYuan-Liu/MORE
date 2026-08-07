suppressPackageStartupMessages({library(MORE)})
NS<-20; NR<-20; NT<-12
S<-paste0("S",1:NS)
reg <- t(sapply(1:NR, function(j) ((0:(NS-1))*7 + (j-1)*13) %% 23 / 23 - 0.5))
rownames(reg)<-paste0("R",1:NR); colnames(reg)<-S
tgt <- t(sapply(1:NT, function(g) {
  a <- 3*reg[((g-1)%%NR)+1,] + 1.5*reg[((g)%%NR)+1,]
  a + 0.05*(((g*5+0:(NS-1))%%11)/11 - 0.5)}))
rownames(tgt)<-paste0("G",1:NT); colnames(tgt)<-S
cond<-data.frame(Ctrl=c(rep(1,NS/2),rep(0,NS/2)),Treat=c(rep(0,NS/2),rep(1,NS/2))); rownames(cond)<-S
assoc<-do.call(rbind,lapply(1:NT,function(g) data.frame(target=paste0("G",g),regulator=paste0("R",1:NR))))
res<-more(targetData=tgt, regulatoryData=list(TF=as.data.frame(reg)), associations=list(TF=assoc),
          condition=cond, method="MLR", varSel="EN", minVariation=c(TF=0), alfa=0.05, seed=123)
for (k in c("G1","G2","G3")) {
  r <- res$ResultsPerTargetF[[k]]
  cat("==", k, "\n")
  cat("  relevantRegulators:", length(r$relevantRegulators), ":", paste(head(r$relevantRegulators,25),collapse=","), "\n")
  cat("  coefficient rows  :", nrow(r$coefficients), ":", paste(head(rownames(r$coefficients),8),collapse=","), "\n")
  ft <- table(r$allRegulators[,"filter"])
  cat("  filter values     :", paste(names(ft),ft,sep="=",collapse=" "), "\n")
}
