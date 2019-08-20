# U ovom modulu se nalaze funkcije koje sluze za skladistenje
# rezultata i varijabli koje su se koristili za dobijanje tih rezultata

import datetime
import json
import os

(dirname, prom) = os.path.split(os.path.dirname(__file__))
json_path = os.path.join(dirname, "Link Prediction\\Results\\results.json")

def create_dict_result(graph_num, pos_size, neg_size, num_of_feat, type_of_feat, func_name):

    result = {}
    now = datetime.datetime.now()
    result["time"] = "{} {}:{}".format(now.date(), now.hour, now.minute)
    result["graph"] = graph_num
    result["accuracy"] = 0
    result["precision_pos"] = 0
    result["precision_neg"] = 0
    result["recall_pos"] = 0
    result["recall_neg"] = 0

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
        json_file["results"].append(result)
    
    with open(json_path, "w") as f:
        json_file = json.dumps(json_file, indent = 4)
        f.write(json_file)
