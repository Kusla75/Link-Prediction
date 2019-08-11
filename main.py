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

# Ne pitajte zasto ovaj deo koda postoji, bitno je da samo otklanja jednu masnu bagcinu
#finale_dataframe = finale_dataframe.loc[:, ~finale_dataframe.columns.str.match('Unnamed')]

print("Dataframes loaded! \n")

feat_to_drop = input("Features to drop: ").split(" ")
if feat_to_drop == [""]:
    feat_to_drop.clear()

type_of_feat = resource_folder.split("_")[1]
result = create_dict_result(graph_num, df_positive.shape[0], df_negative.shape[0], 
    feat_to_drop, df_negative.shape[1] - 1, type_of_feat, LogisticReg.__name__)

# Ako se koriste metrike predvidjanja veza onda ce u results.json
# fajl biti upisani i koeficijenti feature-a
if "ef" in resource_folder:
    return_coef = True
else:
    return_coef = False

score, coef = LogisticReg(finale_dataframe, "CLASS", result["cv_split"], feat_to_drop, return_coef)

result["accuracy"] = round(score, 4)
result["feat_coef"] = coef

document_result(result, json_path)
