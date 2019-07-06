import networkx as nx
from random import randint
import pandas as pd
import os

def recursive_func(graph, ls, step):
	rand_index = randint(0, len(ls) - 1)
	node = ls[rand_index]
	if step != 1:
		step -= 1
		return recursive_func(graph, list(graph.neighbors(node)), step)
	else:
		return node
	
def get_rand_close_nodes(graph, step, n):
	node_pairs = []
	for pom in range(n):
		rand_index = randint(0, len(list(graph.nodes)) - 1)
		n1 = list(graph.nodes)[rand_index]
		node_pairs.append([n1, recursive_func(graph, list(graph.neighbors(n1)), step)]) 
		
	return node_pairs

def df_with_negative_class(graph, step, n):
	node_pairs = get_rand_close_nodes(graph, step, n)
	features_list = list(list(graph.nodes.data())[0][1].keys())
	df = pd.DataFrame(columns = features_list.insert(0, "CLASS"), index = None)
	features_list.remove("CLASS")

	for n1, n2 in node_pairs:
		row = {"CLASS": 0}
		for key in features_list:
			if graph.nodes[n1][key] == graph.nodes[n2][key]:
				row[key] = 1
			else:
				row[key] = 0

		df = df.append(row, ignore_index = True)
	
	for column in df.columns:
		df[column] = df[column].astype(int)

	return df

def df_with_positive_class(graph, n):
	features_list = list(list(graph.nodes.data())[0][1].keys())
	df = pd.DataFrame(columns = features_list.insert(0, "CLASS"), index = None)
	features_list.remove("CLASS")
	rand_edges = []

	for pom in range(n):
		rand = randint(0, len(list(graph.edges)) - 1)
		rand_edges.append(graph.edges[rand])
	
	print(rand_edges)

	for edges in rand_edges:
		row = {"CLASS": 0}
		for key in features_list:
			if graph.nodes[edges[0]][key] == graph.nodes[edges[1]][key]:
				row[key] = 1
			else:
				row[key] = 0
		
		df = df.append(row, ignore_index = True)
	
	for column in df.columns:
		df[column] = df[column].astype(int)

	return df

graph = nx.Graph()

(dirname, prom) = os.path.split(os.path.dirname(__file__))
graph = nx.read_gml(os.path.join(dirname, "Resources\\ego-facebook\\ego-facebook_107.gml"))
print("Graph loaded!")

df_positive = df_with_negative_class(graph, 2, 500)
print("Positive \n")
df_negative = df_with_positive_class(graph, 500)

df_positive.to_csv(os.path.join(dirname, "Resources\\ego-facebook\\positive_feat.csv"), index=False)
df_negative.to_csv(os.path.join(dirname, "Resources\\ego-facebook\\negative_feat.csv"), index=False)
print("CSV files created! \n")


		
