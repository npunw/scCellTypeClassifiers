# scCellTypeClassifiers
A comparison of SVM and Random Forests performance as single-cell cell type classifiers based on simulated data

PMCC_Preprocess.R writes csv data from R data provided by:
https://cf.10xgenomics.com/samples/cell/pbmc68k_rds/pbmc68k_data.rds
https://cf.10xgenomics.com/samples/cell/pbmc68k_rds/all_pure_select_11types.rds

PMCC-SVM.py and PMCC-RF.py were preliminary scripts to test the performance of adjusting kernels (SVM) and number of trees (Random Forests)
PMCCC-SVM_vs_RF.py compares the best versions of both classifiers using Cohen's kappa and F1 beta scores.
