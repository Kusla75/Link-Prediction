# Ova skripta se koristi za spajanje svih grafova

import networkx as nx
from ego_facebook_module import *
import time
import os

(dirname, prom) = os.path.split(os.path.dirname(__file__))
ego_indexes = [0, 107, 348, 414, 686, 698, 1684, 1912, 3437, 3980]

graph = nx.Graph()

graph_save_path = os.path.join(dirname, 
    "Resources\\ego-facebook_merged\\{}\\ego-facebook_merged.gml")
combined_edges_path = os.path.join(dirname,
    "Raw_datasets\\ego-facebook_merged\\merged_edges.txt")
combined_featnames_path = os.path.join(dirname,
    "Raw_datasets\\ego-facebook_merged\\merged.featnames")

inp = input("Create merged featnames file [y/n]: ")
if inp == "y":
    write_all_featnames_in_file(ego_indexes, combined_featnames_path)
elif inp == "n":
    pass
else:
    print("Nije uneta dobra vrednost!")
    exit()

# Učitavaju se sve veze nodova grafa
with open(combined_edges_path, "r") as dataset:
    for line in dataset:
        num = line.strip("\n").split(" ")
        graph.add_edge(num[0], num[1])
    
    print("Combined graph edges loaded! ")

feat_extractor = input("Feat extractor: ")
if feat_extractor == "fbf":
    for index in ego_indexes:
        feat = os.path.join(dirname, 
            "Raw_datasets\\ego-facebook\\{}.feat".format(index))
        featnames = os.path.join(dirname, 
            "Raw_datasets\\ego-facebook\\{}.featnames".format(index))

        graph = extract_feat_by_feat(graph, feat, featnames)

    graph = fill_other_features(graph, combined_featnames_path)
    print("Node attributes loaded!\n")
    nx.write_gml(graph, graph_save_path.format("fbf"))

elif feat_extractor == "ef":
    graph = calculate_edge_features(graph)
    print("Edge attributes loaded!\n")
    nx.write_gml(graph, graph_save_path.format("ef"))

else:
    print("Nije uneta dobra vrednost!")
    exit()

