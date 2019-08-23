import pandas as pd
import os
from ML_algorithms import LogisticReg, KNN, RandomForest
from module import *

(dirname, prom) = os.path.split(os.path.dirname(__file__))

resource_folder = input("Resource folder: ")
graph_num = input("Graph number: ")
size = int(input("Sample size: "))

df_positive = pd.read_csv(os.path.join(dirname, 
    "Link Prediction\\Resources\\{}\\positive_feat_{}.csv".format(resource_folder, graph_num)), nrows = size)
df_positive.dropna(1, inplace = True) 

df_negative = pd.read_csv(os.path.join(dirname, 
    "Link Prediction\\Resources\\{}\\negative_feat_{}.csv".format(resource_folder, graph_num)), nrows = size)
df_negative.dropna(1, inplace = True)

finale_dataframe = pd.concat([df_positive, df_negative])
print("Dataframes loaded! \n")

if graph_num == "merged":
    type_of_feat = resource_folder.split("_")[2]
else:
    type_of_feat = resource_folder.split("_")[1]
    
clf_models = [RandomForest]

for model in clf_models:
    result = create_dict_result(graph_num, df_positive.shape[0], df_negative.shape[0], 
        df_negative.shape[1] - 1, type_of_feat, model.__name__)

    scores = model(finale_dataframe, "CLASS", result["cv_split"])

    result["accuracy"] = round(scores["accuracy"], 4)
    result["precision_pos"] = round(scores["pre_pos"], 4)
    result["precision_neg"] = round(scores["pre_neg"], 4)
    result["recall_pos"] = round(scores["rec_pos"], 4)
    result["recall_neg"] = round(scores["rec_neg"], 4)

    document_result(result, json_path)
    print("{} training finished".format(model.__name__))
