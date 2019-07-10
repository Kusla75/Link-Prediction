# U ovom fajlu se nalaze sve funkcije koje pozivaju
# machine learning algoritme

import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression

def LogisticReg(dataframe, label, ratio, feat_to_drop = []):
    '''Funkcija koja poziva logisticku regresiju. Prvi argument je sam
    dataset nad kojim se trenira, drugi naziv kolone koja oznacava klasu, 
    a treci odnos testa i treninga. Parametar feat_to_drop je tuple featurea
    koji oznacava koliko featurea treba ignorisati.
    '''

    if len(feat_to_drop) != 0:
        dataframe = dataframe.drop(list(feat_to_drop), 1)

    y = np.array(dataframe[label])

    x = np.array(dataframe.drop([label], 1))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = ratio)
    logmodel = LogisticRegression()
    logmodel.fit(x_train, y_train)

    scores = cross_val_score(logmodel, x_test, y_test)
    average_score = sum(scores) / len(scores)
        
    return average_score
