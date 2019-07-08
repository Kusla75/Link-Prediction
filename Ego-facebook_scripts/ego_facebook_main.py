# U ovoj skripti se kreiraju .csv fajlovi
# koji ce se koristiti za treniranje klasifikatora

import networkx as nx
import pandas as pd
import os
from ego_facebook_module import *

graph = nx.Graph()

(dirname, prom) = os.path.split(os.path.dirname(__file__))

resource_folder = input("Resource folder: ")
graph_num = input("Graph number: ")

graph = nx.read_gml(os.path.join(dirname, 
    "Resources\\{}\\ego-facebook_{}.gml".format(resource_folder, graph_num)))
print("Graph loaded! \n")
size = int(input("Size of datasets: "))
step = int(input("Step: "))
print()

df_negative = df_with_negative_class(graph, step, size)
print("Negative dataframe created! \n")
df_positive = df_with_positive_class(graph, size)
print("Positive dataframe created! \n")

df_positive.to_csv(os.path.join(dirname, 
    "Resources\\{}\\positive_feat_{}.csv".format(resource_folder, graph_num)), index=False)
df_negative.to_csv(os.path.join(dirname, 
    "Resources\\{}\\negative_feat_{}.csv".format(resource_folder, graph_num)), index=False)
print("CSV files created! \n")