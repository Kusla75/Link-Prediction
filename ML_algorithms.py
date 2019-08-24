# U ovom fajlu se nalaze sve funkcije koje pozivaju
# machine learning algoritme

import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score
from sklearn.model_selection import cross_validate, train_test_split, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn import neighbors
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from module import settings_path, validation_path, document_result, calculate_validation_values, plot_important_features

scorers = {
    "accuracy" : make_scorer(accuracy_score),
    "pre_pos" : make_scorer(precision_score),
    "rec_pos" : make_scorer(recall_score),
}


def LogisticReg(dataframe, label, cv_split):
    '''Funkcija koja poziva logisticku regresiju. Prvi argument je sam
    dataset nad kojim se trenira, drugi naziv kolone koja oznacava klasu, 
    a treci odnos na koliko delova se deli dataset kada se radi CV. 
    '''

    dataframe = shuffle(dataframe, random_state = 42)
    
    y = np.array(dataframe[label])

    x = np.array(dataframe.drop([label], 1))

    model = LogisticRegression(solver = "lbfgs", max_iter = 2500, random_state = 42, n_jobs = -1) 

    results = cross_validate(model, x, y, cv = cv_split, scoring = scorers, n_jobs = -1)
    
    metric_values = {
        "accuracy" : results["test_accuracy"].mean(),
        "pre_pos" : results["test_pre_pos"].mean(),
        "rec_pos" : results["test_rec_pos"].mean(),
    }

    return metric_values

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

        results = cross_validate(model, x, y, cv = cv_split, scoring = ["accuracy"], n_jobs = -1)

        accuracy = results["test_accuracy"].mean()

        dictionary[accuracy] = k
    
    max_accuracy = max(list(dictionary.keys()))
    best_k = dictionary[max_accuracy]
    print("Best k found!")

    model = neighbors.KNeighborsClassifier(n_neighbors = best_k, n_jobs = -1)

    results = cross_validate(model, x, y, cv = cv_split, scoring = scorers, n_jobs = -1)
    
    metric_values = {
        "accuracy" : results["test_accuracy"].mean(),
        "pre_pos" : results["test_pre_pos"].mean(),
        "rec_pos" : results["test_rec_pos"].mean(),
    }

    return metric_values

def RandomForest(dataframe, label, cv_split, *argv):
    '''Funkcija koja poziva random forest algoritam
    Argumenti su isti kao i kod LogisticReg funkcije
    '''

    dataframe = shuffle(dataframe, random_state = 42)

    y = np.array(dataframe[label])

    x = np.array(dataframe.drop([label], 1))

    if argv[0]:
        random_params = {
            'n_estimators': [100, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000],
            'max_features': ['auto', 'sqrt'],
            'max_depth': [10, 20, 40, 60, 80, 100],
            'min_samples_split': [2, 4, 6, 8, 10],
            'min_samples_leaf': [1, 2, 4, 6, 8],
            'bootstrap': [True, False]
        }
        random_params["max_depth"].append(None)

        random_model = RandomForestClassifier(random_state = 42, n_jobs = -1)

        rf_random_models = RandomizedSearchCV(estimator = random_model, param_distributions = random_params, 
            scoring = scorers, n_iter = 100, cv = 5, verbose = 1, refit = 'accuracy', 
            random_state = 42, n_jobs = -1)

        rf_random_models.fit(x, y)
        cv_results = rf_random_models.cv_results_
        dict_values = calculate_validation_values(cv_results, cv_split)
        document_result(dict_values, validation_path)

        best_params = rf_random_models.best_params_
        best_params['ml_algorithm'] = RandomForest.__name__
        document_result(best_params, settings_path)

        base_model = RandomForestClassifier(n_estimators = best_params['n_estimators'], 
            max_features = best_params['max_features'], max_depth = best_params['max_depth'], 
            min_samples_split = best_params['min_samples_split'], min_samples_leaf = best_params['min_samples_leaf'],
            bootstrap = best_params['bootstrap'], random_state = 42, n_jobs = -1)
    else:
        base_model = RandomForestClassifier(
            n_estimators = 1600,
            min_samples_split = 8,
            min_samples_leaf = 1,
            max_features = "auto",
            max_depth = None,
            bootstrap = False, random_state = 42, n_jobs = -1)

    results = cross_validate(base_model, x, y, cv = cv_split, scoring = scorers, n_jobs = -1)

    if argv[1]:
        column_names = dataframe.drop([label], 1).columns
        print(list(column_names))
        plot_important_features(results, column_names)
    
    metric_values = {
        "accuracy" : results["test_accuracy"].mean(),
        "pre_pos" : results["test_pre_pos"].mean(),
        "rec_pos" : results["test_rec_pos"].mean(),
    }

    return metric_values

