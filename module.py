# U ovom modulu se nalaze funkcije koje sluze za skladistenje
# rezultata i varijabli koje su se koristili za dobijanje tih rezultata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import json
import os

(dirname, prom) = os.path.split(os.path.dirname(__file__))
json_path = os.path.join(dirname, "Link Prediction\\Results\\results.json")
settings_path = json_path.replace('results', 'settings')
validation_path = json_path.replace('results', 'validation')

def create_dict_result(graph_num, pos_size, neg_size, num_of_feat, type_of_feat, func_name):

    result = {}
    now = datetime.datetime.now()
    result["time"] = "{} {}:{}".format(now.date(), now.hour, now.minute)
    result["graph"] = graph_num
    result["accuracy"] = 0
    result["precision_pos"] = 0
    result["recall_pos"] = 0

    result["cv_split"] = 5
    result["positive_size"] = pos_size
    result["negative_size"] = neg_size

    result["num_of_feat"] = num_of_feat
    result["type_of_feat"] = type_of_feat
    result["ML_algorithm"] = func_name

    return result

def document_result(result, json_path):
    with open(json_path, "r") as f:
        json_file = json.load(f)
        if "settings" in json_path:
            json_file["settings"].append(result)
        elif "validation" in json_path:
            json_file["validation"].append(result)
        else:
            json_file["results"].append(result)
    
    with open(json_path, "w") as f:
        json_file = json.dumps(json_file, indent = 4)
        f.write(json_file)

def calculate_validation_values(cv_results, cv_split):
    accuracy, pre_pos, rec_pos = [], [], []

    for i in range(cv_split):
        accuracy.append(cv_results["split{}_test_accuracy".format(i)].mean())
        pre_pos.append(cv_results["split{}_test_pre_pos".format(i)].mean())
        rec_pos.append(cv_results["split{}_test_rec_pos".format(i)].mean())
    
    dict_values = {
        "accuracy" : sum(accuracy)/len(accuracy),
        "precision_pos" : sum(pre_pos)/len(pre_pos),
        "recall_peg" : sum(rec_pos)/len(rec_pos)
    }

    return dict_values

def plot_important_features(results, column_names):
    index_tuple = np.where(results["test_accuracy"] == np.amax(results["test_accuracy"]))
    best_est_index = int(index_tuple[0])

    best_estimator = results['estimator'][best_est_index]

    feature_importances = best_estimator.feature_importances_

    if len(feature_importances) < 50:
        print(feature_importances)
        series = pd.Series(feature_importances, index = column_names)
        series.plot(kind = 'barh')
        plt.show()
    else:
        column_names = list(column_names)
        feat_imp_dict = {}
        for i in range(len(column_names)):
            feat_imp_dict[column_names[i]] = feat_imp_dict[i]

        print()
        