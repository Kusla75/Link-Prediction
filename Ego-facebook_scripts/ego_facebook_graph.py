# U ovoj skripti se ucitavaju podaci iz ego-facebook 
# dataseta i upisuju u .gml fajl

import networkx as nx
from ego_facebook_module import *
import time
import os

(dirname, prom) = os.path.split(os.path.dirname(__file__))

graph_num = input("Graph number: ")
feat_extractor = input("Feat extractor: ")

edges_path = os.path.join(dirname, 
    "Raw_datasets\\ego-facebook\\{}.edges".format(graph_num))
feat_path = edges_path.replace(".edges", ".feat")
featnames_path = edges_path.replace(".edges", ".featnames")
graph_path = os.path.join(dirname, 
    "Resources\\ego-facebook_{}\\ego-facebook_{}.gml".format(feat_extractor, graph_num))

graph = nx.Graph()

# Ucitavanje veza izmedju nodova
with open(edges_path, "r") as dataset:
    for line in dataset:
        num = line.strip("\n").split(" ")
        graph.add_edge(num[0], num[1])

    print("Edges loaded! \n")

# Korsinik na osnovu unosa naznacava nacin na koji zeli da 
# se upisu featuri u graf pozivaju se feature extractori iz modula
# fbf - feature by feature
# gf - grouped features
# ef - edge features
# wef - weighted edge features
if feat_extractor == "fbf" or feat_extractor == "wef":
    graph = extract_feat_by_feat(graph, feat_path)
    if feat_extractor == "wef":
        graph = calculate_edge_features(graph, add_weight = True)
elif feat_extractor == "gf":
    graph = group_features(graph, featnames_path, feat_path)
elif feat_extractor == "ef":
    graph = calculate_edge_features(graph)
else:
    print("Niste uneli dobar feature extractor! ")
    exit()

# Upisivanje graf objekta u .gml fajl
with open(graph_path, "w") as new_file:
    nx.write_gml(graph, graph_path)
    print("Graph saved to " + graph_path + "\n")




