# U ovom fajlu se nalaze sve funkcije koje pozivaju
# machine learning algoritme

import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

def LogisticReg(dataframe, label, cv_split):
    '''Funkcija koja poziva logisticku regresiju. Prvi argument je sam
    dataset nad kojim se trenira, drugi naziv kolone koja oznacava klasu, 
    a treci odnos na koliko delova se deli dataset kada se radi CV. 
    '''

    dataframe = shuffle(dataframe, random_state = 42)
    
    y = np.array(dataframe[label])

    x = np.array(dataframe.drop([label], 1))

    model = LogisticRegression(solver = "lbfgs", max_iter = 2500, random_state = 42, n_jobs = -1) 

    results = cross_validate(model, x, y, cv = cv_split,  
        scoring = ["precision", "recall", "accuracy"], n_jobs = -1)
    
    scores = [
        results["test_accuracy"].mean(),
        results["test_precision"].mean(),
        results["test_recall"].mean()
    ]

    return scores

def KNN(dataframe, label, cv_split):
    '''Funkcija koja poziva k nearest neigbours.
    Argumenti su isti kao i kod LogisticReg funkcije 
    '''

    dataframe = shuffle(dataframe, random_state = 42)

    y = np.array(dataframe[label])

    x = np.array(dataframe.drop([label], 1))
    dictionary = {}

    # ovde se iterira kroz k vrednosti i zapisuju se srednje
    # tačnosti koje klasifikator daje za njih
    for k in range(3, 12):
        model = neighbors.KNeighborsClassifier(n_neighbors = k, n_jobs = -1)

        results = cross_validate(model, x, y, cv = cv_split,
            scoring = ["accuracy"], n_jobs = -1)

        accuracy = results["test_accuracy"].mean()

        dictionary[accuracy] = k
    
    max_accuracy = max(list(dictionary.keys()))
    best_k = dictionary[max_accuracy]
    print("Best k found!")

    model = neighbors.KNeighborsClassifier(n_neighbors = best_k, n_jobs = -1)
    results = cross_validate(model, x, y, cv = cv_split,
            scoring = ["precision", "recall", "accuracy"], n_jobs = -1)
    
    scores = [
        results["test_accuracy"].mean(),
        results["test_precision"].mean(),
        results["test_recall"].mean()
    ]
    
    return scores

def RandomForest(dataframe, label, cv_split):
    '''Funkcija koja poziva random forest algoritam
    Argumenti su isti kao i kod LogisticReg funkcije
    '''

    dataframe = shuffle(dataframe, random_state = 42)

    y = np.array(dataframe[label])

    x = np.array(dataframe.drop([label], 1))

    model = RandomForestClassifier(n_estimators = 100, oob_score = True, random_state = 42, n_jobs = -1)

    results = cross_validate(model, x, y, cv = cv_split, 
        scoring = ["precision", "recall", "accuracy"], n_jobs = -1)

    scores = [
        results["test_accuracy"].mean(),
        results["test_precision"].mean(),
        results["test_recall"].mean()
    ]
    
    return scores
