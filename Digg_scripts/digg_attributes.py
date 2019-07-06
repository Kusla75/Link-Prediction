# Ova skripta se koristi za učitavanje .gml fajla u graph objekat.
# Namenjena je za digg dataset
# Od tog grafa kreira se .csv fajl sa atributima za svaku vezu grafa
# Kreiraju se dva fajla onaj sa pravim i onaj sa pseudo atributima

import os
from metric_functions import *
import time
import networkx as nx
import pandas as pd
from random import randint

(dirname, prom) = os.path.split(os.path.dirname(__file__))
attr_fname = os.path.join(dirname, "Resources/Digg/digg_attributes.csv")
pseudo_fname = os.path.join(dirname, "Resources/Digg/digg_pseudo_attributes.csv")
graph = nx.Graph()
graph = nx.read_gml(os.path.join(dirname, "Resources/Digg/digg_graph.gml"))

print("Graph loaded! \n")

# Ovde se kreiraju pravi atributi i upisuju u .csv fajl
if not os.path.isfile(attr_fname):
    with open(attr_fname, "w") as f:
        t_start = time.time()
        f.write("TIMESTAMP,REAL,CN,JC,PA,AA,RA\n")

        for edge in graph.edges.data():
            cn = common_neighbors(graph, edge[0], edge[1])
            jc = round(jaccards_coefficient(graph, edge[0], edge[1]), 3)

            if cn == 0:     # Ovde se filtriraju "losi" podaci koji nece 
                continue    # doprineti treniranju algoritma

            pa = preferential_attachment(graph, edge[0], edge[1])
            aa = round(adamic_adar(graph, edge[0], edge[1]), 3)
            ra = round(resource_allocation(graph, edge[0], edge[1]), 3)

            f.write("{},{},{},{},{},{},{}\n".format(edge[2]["timestamp"], 1, cn, jc, pa, aa, ra))
                                   
        print("File created!")
        print("Time: ", time.time() - t_start)
        print()
else:
    print("File with attributes already exists! \n")


pseudo_graph = nx.Graph()
nodes = list(graph.nodes)
nodes.sort()

NUM_OF_PSEUDO_LINKS = 6000000     # Sto je broj pseudo veza u grafu veci vrednosti metrika ce biti "bolji"
i = 0

# Ovde se kreiraju pseudo atributi i upisuju u .csv fajl
if not os.path.isfile(pseudo_fname):

    while i <= NUM_OF_PSEUDO_LINKS:
        r1 = randint(0, len(nodes)-1)
        r2 = randint(0, len(nodes)-1)
        if (r1 != r2 and not graph.has_edge(nodes[r1], nodes[r2]) 
            and not pseudo_graph.has_edge(nodes[r1], nodes[r2])):

            pseudo_graph.add_edge(nodes[r1], nodes[r2])
            i += 1
    
    print("Pseudo graph created! \n")

    with open(pseudo_fname, "w") as f:
        f.write("TIMESTAMP,REAL,CN,JC,PA,AA,RA\n")

        for edge in pseudo_graph.edges:
            cn = common_neighbors(graph, edge[0], edge[1])
            if cn == 0:
                continue
                
            jc = round(jaccards_coefficient(graph, edge[0], edge[1]), 5)
            pa = preferential_attachment(graph, edge[0], edge[1])
            aa = round(adamic_adar(graph, edge[0], edge[1]), 5)
            ra = round(resource_allocation(graph, edge[0], edge[1]), 5)

            f.write("{},{},{},{},{},{},{}\n".format(-1, 0, cn, jc, pa, aa, ra))


    print("Pseudo attributes file created! ")

else:
    print("File with pseudo attributes already exists! \n")
