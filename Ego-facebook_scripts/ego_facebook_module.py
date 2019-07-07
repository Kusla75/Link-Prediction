# U ovom modulu se nalaze sve funkcije koje su potrebne za kreiranje
# .csv fajlova koji ce se koristiti za treniranje klasifikatora

import networkx as nx
from random import randint
import pandas as pd
import os

def recursive_func(graph, ls, step):
	'''Rekurzivna funkcija koja na osnovu stepa iterira do nasumicnog noda
		koji je relativno blizu zadatom nodu. Drugim recima ova funkc vraca 
		nod koji je komsija komsije komsije... od zadatkog noda.

		ls - lista komsija zadatkog noda
		step - koliko koraka ce biti udaljeni zadati i nasumicni nod
	'''

	rand_index = randint(0, len(ls) - 1)
	node = ls[rand_index]
	if step != 1:
		step -= 1
		return recursive_func(graph, list(graph.neighbors(node)), step)
	else:
		return node

def get_rand_close_nodes(graph, step, n):
	'''Koriscenjem rekurzivne funkcije vraca par nodova koji su relativno blizu.

		step - definise koliko koraka su ta dva noda udaljena
		n - koliko ce tih parova nodova biti
	'''

	node_pairs = []
	while n != 0:
		rand_index = randint(0, len(list(graph.nodes)) - 1)
		n1 = list(graph.nodes)[rand_index]
		n2 = recursive_func(graph, list(graph.neighbors(n1)), step)
		if not graph.has_edge(n1, n2):
			node_pairs.append([n1, n2])
			n -= 1
		else:
			continue
		
	return node_pairs

def df_with_negative_class(graph, step, n):
	'''Kreira dataframe koji ce da sadrzi feature negativne klase.

		n - definise koliko ce biti redova u samom dataframe-u
	'''

	node_pairs = get_rand_close_nodes(graph, step, n)
	features_list = list(list(graph.nodes.data())[0][1].keys())
	df = pd.DataFrame(columns = features_list.insert(0, "CLASS"), index = None)
	features_list.remove("CLASS")

	for n1, n2 in node_pairs:
		try:
			row = {"CLASS": 0}
			for key in features_list:
				if graph.nodes[n1][key] == graph.nodes[n2][key]:
					row[key] = 1
				else:
					row[key] = 0

			df = df.append(row, ignore_index = True)
		except KeyError:
			print("Dogodio se KeyError!")
			continue
				
	return df

def df_with_positive_class(graph, n):
	'''Kreira dataframe koji ce da sadrzi feature pozitivne klase.

		n - definise koliko ce biti redova u samom dataframe-u
	'''

	features_list = list(list(graph.nodes.data())[0][1].keys())
	df = pd.DataFrame(columns = features_list.insert(0, "CLASS"), index = None)
	features_list.remove("CLASS")
	rand_edges = []
	rand_indexes = []

	while n != 0:
		rand = randint(0, len(list(graph.edges)) - 1)
		if rand in rand_indexes:
			continue
		else:
			rand_edges.append(list(graph.edges)[rand])
			rand_indexes.append(rand)
			n -= 1
	
	for edges in rand_edges:
		try:
			row = {"CLASS": 1}
			for key in features_list:
				if graph.nodes[edges[0]][key] == graph.nodes[edges[1]][key]:
					row[key] = 1
				else:
					row[key] = 0
			
			df = df.append(row, ignore_index = True)
		except KeyError:
			print("Dogodio se KeyError!")
			continue
	
	return df
