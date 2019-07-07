import pandas as pd
import os
from ML_algorithms import *
from module import *

(dirname, prom) = os.path.split(os.path.dirname(__file__))

graph_num = input("Graph number: ")
size = int(input("Sample size: "))

df_positive = pd.read_csv(os.path.join(dirname, 
    "Link Prediction\\Resources\\ego-facebook\\positive_feat_{}.csv".format(graph_num))).sample(size)

df_negative = pd.read_csv(os.path.join(dirname, 
    "Link Prediction\\Resources\\ego-facebook\\negative_feat_{}.csv".format(graph_num))).sample(size)

finale_dataframe = pd.concat([df_positive, df_negative])
print("Dataframes loaded! \n")

result = create_dic_result(int(graph_num), df_positive.shape[0], 
    df_negative.shape[0], df_negative.shape[1] - 1, LogisticReg.__name__)

score = LogisticReg(finale_dataframe, "CLASS", result["test_size"])

result["accuracy"] = round(score, 4)
document_result(result, json_path)
