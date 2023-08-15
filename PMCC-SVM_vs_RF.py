# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 19:19:52 2023

@author: neil_
"""


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import fbeta_score
import matplotlib.pyplot as plt
import time
scaler = StandardScaler()


os. chdir('C:\\Users\\neil_\\Desktop\\PMCC Techincal\\GitHub Files') 

reference_matrix = pd.read_csv("reference_matrix.csv", index_col=0)
num_reps = 600
num_feats = num_reps #this is because there are 11 classes. Replicating
#the simulated data will have 11 x n repetitions. If we set the number of
#features (i.e., genes) to be n, the sample size will always be one 
#order of magnitude greater than the number of features (recommended)

#Normalize reference matrix
scaler = StandardScaler()
ref_mat_norm = scaler.fit_transform(reference_matrix)
ref_df_norm = pd.DataFrame(ref_mat_norm, columns=reference_matrix.columns, index=reference_matrix.index)

#Randomly sample genes to be used as features
np.random.seed(1234)
ref_samp = ref_df_norm.sample(num_feats)



# Initialize lists to store Cohen Kappa scores
svm_scores = []
rf_scores = []
svm_f1s =[]
rf_f1s =[]
svm_times = []
rf_times = []

# Loop to repeat tests with varying noise factors
noisy_factors = np.linspace(0, 3, 11)
i=0
for noisy_factor in noisy_factors:
    np.random.seed(i)
    i += 1
    # Replicate and add variation to reference subsample to generate
    # simulated data
    X_full = pd.DataFrame(np.repeat(ref_samp.values, num_reps, axis=1))
    X_full += np.random.normal(0, noisy_factor, size=X_full.shape)  # Add some noise
    y_full = np.repeat(np.arange(ref_samp.shape[1]), num_reps)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_full.T, y_full, test_size=0.2, random_state=42)

    svm_sta = time.time()
    # Train the SVM classifier with rbf kernel (recommended)
    svm_classifier = SVC(kernel='rbf')
    svm_classifier.fit(X_train, y_train)
    svm_fin = time.time()
    svm_ela = svm_fin-svm_sta
    svm_times.append(svm_ela)
    
    # Make predictions on the test set
    y_pred_svm = svm_classifier.predict(X_test)

    # Calculate Cohen Kappa Score
    svm_scores.append(cohen_kappa_score(y_test, y_pred_svm))
    
    #calculate f1 score
    f_beta_scores = []
    for class_i in range(10):
        f_beta = fbeta_score(y_test == class_i, y_pred_svm == class_i, beta=1)
        f_beta_scores.append(f_beta)
    svm_f1s.append(np.mean(f_beta_scores))

    # Train the RF classifier, make predications and calculate accuracy/Cohen Kappa
    rf_sta = time.time()
    rf_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
    rf_classifier.fit(X_train, y_train)
    rf_fin = time.time()
    rf_ela = rf_fin-rf_sta
    rf_times.append(rf_ela)
    
    y_pred_rf = rf_classifier.predict(X_test)

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_scores.append(cohen_kappa_score(y_test, y_pred_rf))
    
    #calculate f1 score
    f_beta_scores = []
    for class_i in range(10):
        f_beta = fbeta_score(y_test == class_i, y_pred_rf == class_i, beta=1)
        f_beta_scores.append(f_beta)
    rf_f1s.append(np.mean(f_beta_scores))
    
max_svm_ela = str(round(svm_times[4],2))
max_rf_ela = str(round(rf_times[4],2))

plt.plot(noisy_factors, svm_scores, label='SVM (' + max_svm_ela + 's)')
plt.plot(noisy_factors, rf_scores, label='Random Forest (' + max_rf_ela + 's)')
plt.axhline(y=0.4, color='gray', linestyle='--', label=None)
plt.xlabel('Noise Factor')
plt.ylabel('Cohen Kappa Score')
plt.title('Reliability of single-cell cell type classifiers')
plt.legend(title='Classifier, (execution time)')
plt.grid(True)
plt.show()

plt.plot(noisy_factors, svm_f1s, label='SVM (' + max_svm_ela + 's)')
plt.plot(noisy_factors, rf_f1s, label='Random Forest (' + max_rf_ela + 's)')
plt.axhline(y=0.5, color='gray', linestyle='--', label=None)
plt.xlabel('Noise Factor')
plt.ylabel('F1 beta scores')
plt.title('Precision-recall performance of single-cell cell type classifiers')
plt.legend(title='Classifier, (execution time)')
plt.grid(True)
plt.show()