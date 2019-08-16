# U ovom fajlu se nalaze sve funkcije koje pozivaju
# machine learning algoritme

import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors, svm

def LogisticReg(dataframe, label, cv_split, feat_to_drop = []):
    '''Funkcija koja poziva logisticku regresiju. Prvi argument je sam
    dataset nad kojim se trenira, drugi naziv kolone koja oznacava klasu, 
    a treci odnos na koliko delova se deli dataset kada se radi CV. 
    Parametar feat_to_drop je tuple featurea koji oznacava koliko featurea treba ignorisati.
    '''

    if len(feat_to_drop) != 0:
        dataframe = dataframe.drop(list(feat_to_drop), 1)

    y = np.array(dataframe[label])

    x = np.array(dataframe.drop([label], 1))

    model = LogisticRegression(solver = "lbfgs", max_iter = 2500, random_state = 42, n_jobs = -1) 

    results = cross_validate(model, x, y, cv = cv_split,  
        scoring = ["precision_macro", "recall_macro", "accuracy"], n_jobs = -1)
    
    scores = [
        results["test_accuracy"].mean(),
        results["test_precision_macro"].mean(),
        results["test_recall_macro"].mean()
    ]

    return scores

def KNN(dataframe, label, cv_split, feat_to_drop = []):

    if len(feat_to_drop) != 0:
        dataframe = dataframe.drop(list(feat_to_drop), 1)

    y = np.array(dataframe[label])

    x = np.array(dataframe.drop([label], 1))

    model = neighbors.KNeighborsClassifier(n_jobs = -1)

    results = cross_validate(model, x, y, cv = cv_split,
        scoring = ["precision_macro", "recall_macro", "accuracy"], n_jobs = -1)

    scores = [
        results["test_accuracy"].mean(),
        results["test_precision_macro"].mean(),
        results["test_recall_macro"].mean()
    ]

    return scores

def SVM(dataframe, label, cv_split, feat_to_drop = []):

    if len(feat_to_drop) != 0:
        dataframe = dataframe.drop(list(feat_to_drop), 1)

    y = np.array(dataframe[label])

    x = np.array(dataframe.drop([label], 1))

    model = svm.SVC(random_state = 42)

    results = cross_validate(model, x, y, cv = cv_split,
        scoring = ["precision_macro", "recall_macro", "accuracy"], n_jobs = -1)

    scores = [
        results["test_accuracy"].mean(),
        results["test_precision_macro"].mean(),
        results["test_recall_macro"].mean()
    ]

    return scores