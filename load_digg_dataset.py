# Ova skripta se koristi za učitavanje digg dataseta
# i kreiranje graph objekta za taj dataset

import networkx as nx
import time
import metric_functions as mf
import os

graph = nx.Graph()
dataset_path = ".\\Raw Datasets\\digg_dataset.csv"
graph_path = ".\\Graphs\\digg_graph.gml"

# Učitavanje nodova i veza iz dataseta i upisivanje u graf
# na svaku upisanu vezu dodaje se timestamp vrednost
with open(dataset_path, "r") as dataset:
    dataset.readline()
    for line in dataset:
        num = line.split(",")
        graph.add_edge(int(num[2]), int(num[3]), timestamp = int(num[1]))

    print("Graph loaded! \n")

with open(graph_path, "w") as new_file:
    nx.write_gml(graph, graph_path)
    print("Graph saved to " + graph_path)

