# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 12:31:53 2023

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
rf10_scores = []
rf50_scores = []
rf100_scores = []

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


    # Train the SVM classifier with linear kernel
    rf_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 42)
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_svm = rf_classifier.predict(X_test)

    # Calculate Cohen Kappa Score
    rf10_scores.append(cohen_kappa_score(y_test, y_pred_svm))
    # #cv_scores = cross_val_score(SVC(), train[features], train[target], cv=5)

    # Train the SVM classifier with rbf kernel (recommended)
    rf_classifier = RandomForestClassifier(n_estimators = 50, criterion = 'entropy', random_state = 42)
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_svm = rf_classifier.predict(X_test)

    # Calculate Cohen Kappa Score
    rf50_scores.append(cohen_kappa_score(y_test, y_pred_svm))


    # Train the RF classifier with n_est =100
    rf_classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 42)
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred_svm = rf_classifier.predict(X_test)

    # Calculate Cohen Kappa Score
    rf100_scores.append(cohen_kappa_score(y_test, y_pred_svm))

#plot Cohen's kappa across all three n_estimators
plt.plot(noisy_factors, rf10_scores, label='10')
plt.plot(noisy_factors, rf50_scores, label='50')
plt.plot(noisy_factors, rf100_scores, label='100')
plt.axhline(y=0.4, color='gray', linestyle='--')
plt.xlabel('Noise Factor')
plt.ylabel('Cohen Kappa Score')
plt.title('Reliability of Random Forest')
plt.legend(title = "n_estimators, Number of Trees")
plt.grid(True)
plt.show()