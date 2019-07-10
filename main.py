import pandas as pd
import os
from ML_algorithms import *
from module import *

(dirname, prom) = os.path.split(os.path.dirname(__file__))

resource_folder = input("Resource folder: ")
graph_num = input("Graph number: ")
size = int(input("Sample size: "))

df_positive = pd.read_csv(os.path.join(dirname, 
    "Link Prediction\\Resources\\{}\\positive_feat_{}.csv".format(resource_folder, graph_num))).head(size)

df_negative = pd.read_csv(os.path.join(dirname, 
    "Link Prediction\\Resources\\{}\\negative_feat_{}.csv".format(resource_folder, graph_num))).head(size)

finale_dataframe = pd.concat([df_positive, df_negative])
print("Dataframes loaded! \n")

feat_to_drop = input("Features to drop: ").split(" ")
if feat_to_drop == [""]:
    feat_to_drop.clear()

type_of_feat = resource_folder.split("_")[1]
result = create_dict_result(int(graph_num), df_positive.shape[0], df_negative.shape[0], 
    feat_to_drop, df_negative.shape[1] - 1, type_of_feat, LogisticReg.__name__)

score = LogisticReg(finale_dataframe, "CLASS", result["test_size"], feat_to_drop)

result["accuracy"] = round(score, 4)
document_result(result, json_path)
