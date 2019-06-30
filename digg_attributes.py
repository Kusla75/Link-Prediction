# Ova skripta se koristi za učitavanje .gml fajla u graph objekat.
# Namenjena je za digg dataset
# Od tog grafa kreira se .csv fajl sa atributima za svaku vezu grafa
# .csv fajl sadrži i pseudo atribute

import os
import metric_functions as mf
import time
import networkx as nx

path = ".\\Resources\\Digg"
graph = nx.Graph()
graph = nx.read_gml(path + "\\digg_graph.gml")

print("Graph loaded! \n")

if not os.path.isfile(path + "\\digg_attributes.csv"):
    with open(path + "\\digg_attributes.csv", "w") as f:
        t_start = time.time()
        f.write("TIMESTAMP,REAL,CN,JC,PA,AA,RA\n")

        for edge in graph.edges.data():
            cn = mf.common_neighbors(graph, edge[0], edge[1])
            jc = round(mf.jaccards_coefficient(graph, edge[0], edge[1]), 3)

            if cn == 0:     # Ovde se filtriraju "losi" podaci koji nece 
                continue    # doprineti treniranju algoritma

            pa = mf.preferential_attachment(graph, edge[0], edge[1])
            aa = round(mf.adamic_adar(graph, edge[0], edge[1]), 3)
            ra = round(mf.resource_allocation(graph, edge[0], edge[1]), 3)

            f.write("{},{},{},{},{},{},{}\n".format(edge[2]["timestamp"], 1, cn, jc, pa, aa, ra))
                       
        print("File created!")
        print("Time: ", time.time() - t_start)
        print()
else:
    print("File with attributes exists!")

