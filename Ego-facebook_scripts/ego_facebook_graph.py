import networkx as nx
import time
import os

(dirname, prom) = os.path.split(os.path.dirname(__file__))

graph_num = input("Graph number: ")

edges_path = os.path.join(dirname, 
    "Raw_datasets\\ego-facebook\\{}.edges".format(graph_num))
feat_path = os.path.join(dirname, 
    "Raw_datasets\\ego-facebook\\{}.feat".format(graph_num))
graph_path = os.path.join(dirname, 
    "Resources\\ego-facebook\\ego-facebook_{}.gml".format(graph_num))

graph = nx.Graph()

# Ucitavanje veza izmedju nodova
with open(edges_path, "r") as dataset:
    for line in dataset:
        num = line.strip("\n").split(" ")
        graph.add_edge(num[0], num[1])

    print("Edges loaded! \n")

# Ucitavanje featura za nodove
with open(feat_path, "r") as dataset:
    dic = {}
    for line in dataset:
        dic.clear()
        features = line.strip("\n").split(" ")
        node = features[0]
        dic = {node : {}}
        del features[0]
        for i in range(len(features)):
            dic[node]["F" + str(i)] = features[i]
        nx.set_node_attributes(graph, dic)
                
# Upisivanje graf objekta u .gml fajl
with open(graph_path, "w") as new_file:
    nx.write_gml(graph, graph_path)
    print("Graph saved to " + graph_path + "\n")




