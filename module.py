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
grouped_featnames_file = os.path.join(dirname, "Link Prediction\\ego-fb_all_featnames")

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

def plot_important_features(estimator, column_names):
    feature_importances = estimator.feature_importances_

    column_names = list(column_names)
    feat_imp_dict = {}
    best_feat_names = []
    best_feat_values = []
    dic = {}

    for i in range(len(column_names)):
        feat_imp_dict[column_names[i]] = feature_importances[i]

    for i in range(20):
        values = list(feat_imp_dict.values())
        keys = list(feat_imp_dict.keys())

        max_key = keys[values.index(max(values))]
        best_feat_names.append(max_key)
        best_feat_values.append(feat_imp_dict[max_key])

        del feat_imp_dict[max_key]
    
    for i in range(len(best_feat_names)):
        dic[best_feat_names[i]] = best_feat_values[i]

    print(dic) 
    series = pd.Series(best_feat_values, index = best_feat_names)
    series.plot(kind = 'barh')
    plt.show()

def plot_grouped_node_feat(estimator, column_names):
    feature_importances = estimator.feature_importances_
    grp_featnames = []
    column_names = list(column_names)
    feat_imp_dict = {}
    dic = {}
    # feat_values = []
    # feat_names = []

    with open(grouped_featnames_file, "r") as featnames_file:
        for line in featnames_file:
            grp_featnames.append(line.strip("\n"))

    for i in range(len(column_names)):
        feat_imp_dict[column_names[i].replace("F", "")] = feature_importances[i]

    for elem in grp_featnames:
        feat_grp = elem.split(" ")[0]
        dic[feat_grp] = 0
    
    for key, val in feat_imp_dict.items():
        for elem in grp_featnames:
            feat_grp = elem.split(" ")[0]
            feat_id = elem.split(" ")[1]

            if key == feat_id:
                dic[feat_grp] += val
    
    print(dic) 
    series = pd.Series(list(dic.values()), index = list(dic.keys()))
    series.plot(kind = 'barh')
    plt.show()

    
    
        