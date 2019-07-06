# Ova skripta se koristi za učitavanje digg dataseta
# i kreiranje graph objekta za taj dataset

import networkx as nx
import time
import os

(dirname, prom) = os.path.split(os.path.dirname(__file__))

graph = nx.Graph()
dataset_path = os.path.join(dirname, "Raw_datasets/digg_dataset.csv")
graph_path = os.path.join(dirname, "Resources/Digg/digg_graph.gml")

# Učitavanje nodova i veza iz dataseta i upisivanje u graf
# na svaku upisanu vezu dodaje se timestamp vrednost
t_start = time.time()
with open(dataset_path, "r") as dataset:
    dataset.readline()
    for line in dataset:
        num = line.split(",")
        graph.add_edge(int(num[2]), int(num[3]), timestamp = int(num[1]))

    print("CSV file loaded! \n")

with open(graph_path, "w") as new_file:
    nx.write_gml(graph, graph_path)
    print("Graph saved to " + graph_path + "\n")

print("Time: ", time.time() - t_start)
