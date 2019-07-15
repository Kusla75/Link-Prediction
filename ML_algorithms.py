# U ovom fajlu se nalaze sve funkcije koje pozivaju
# machine learning algoritme

import pandas as pd
import numpy as np
import sklearn as skl
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegressionCV

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

    logmodel = LogisticRegressionCV(cv = 5, random_state = 1, n_jobs = -1).fit(x,y)

#    average_score = cross_val_score(logmodel, x, y, cv = cv_split, n_jobs = -1).mean()
 
    average_score = logmodel.score(x,y)
    print(logmodel.coef_)
    return average_score
