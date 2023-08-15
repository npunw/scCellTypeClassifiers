

#setwd('C:\\Users\\neil_\\Desktop\\PMCC Techincal\\GitHub Files')
source('util.R')
pbmc_68k <- readRDS('./Data/pbmc68k_data.rds')
pure_11 <- readRDS('./Data/all_pure_select_11types.rds')
purified_ref_11 <- load_purified_pbmc_types(pure_11,pbmc_68k$ens_genes)

write.csv(purified_ref_11, file ="reference_matrix.csv", row.names=T)
#write.table(purified_ref_11, "reference_matrix.csv", row.names=F, col.names=F, sep=",")

mean(purified_ref_11[,11])

profile_sd <-  apply(purified_ref_11, 2, sd)
##hist()