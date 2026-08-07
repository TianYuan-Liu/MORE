suppressPackageStartupMessages(library(MORE))
args <- commandArgs(trailingOnly=TRUE)
NT<-as.integer(args[1]); NR<-as.integer(args[2]); NS<-as.integer(args[3]); ND<-as.integer(args[4])
S<-paste0("S",1:NS)
reg <- t(sapply(1:NR, function(j) ((0:(NS-1))*7 + (j-1)*13) %% 23 / 23 - 0.5))
rownames(reg)<-paste0("R",1:NR); colnames(reg)<-S
tgt <- t(sapply(1:NT, function(g){
  a <- rep(0,NS); for (d in 0:(ND-1)) a <- a + (3/(d+1))*reg[((g-1+d)%%NR)+1,]
  a + 0.05*(((g*5+0:(NS-1))%%11)/11-0.5)}))
rownames(tgt)<-paste0("G",1:NT); colnames(tgt)<-S
cond<-data.frame(Ctrl=c(rep(1,NS/2),rep(0,NS/2)),Treat=c(rep(0,NS/2),rep(1,NS/2))); rownames(cond)<-S
assoc<-do.call(rbind,lapply(1:NT,function(g) data.frame(target=paste0("G",g),regulator=paste0("R",1:NR))))
edges<-function(sd){
  r<-more(targetData=tgt, regulatoryData=list(TF=as.data.frame(reg)), associations=list(TF=assoc),
          condition=cond, method="MLR", varSel="EN", minVariation=c(TF=0), alfa=0.05, seed=sd)
  unique(paste(RegulationPerCondition(r)$targetF, RegulationPerCondition(r)$regulator, sep=":::"))}
a<-edges(123); b<-edges(456); c3<-edges(789)
j<-function(x,y) length(intersect(x,y))/length(union(x,y))
cat(sprintf("R-vs-R Jaccard: 123/456=%.4f  123/789=%.4f  456/789=%.4f  MIN=%.4f\n",
            j(a,b), j(a,c3), j(b,c3), min(j(a,b),j(a,c3),j(b,c3))))
