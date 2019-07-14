# U ovom modulu se nalaze sve funkcije koje su potrebne za kreiranje
# .csv fajlova koji ce se koristiti za treniranje klasifikatora

import networkx as nx
from random import randint
import pandas as pd
import sys, os

(dirname, prom) = os.path.split(os.path.dirname(__file__))
sys.path.append(dirname)
from metric_functions import *

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
		if n1 == n2:
			continue

		if not graph.has_edge(n1, n2) and not [n1, n2] in node_pairs:
			node_pairs.append([n1, n2])
			n -= 1
		else:
			continue
		
	return node_pairs

def negative_class_node_feat(graph, step, n):
	'''Kreira dataframe koji ce da sadrzi feature negativne klase.
		Featuri su preuzeti od nodova

		step - koliko ce nasumisni nodovi biti blizu
		n - definise koliko ce biti redova u samom dataframe-u
	'''

	node_pairs = get_rand_close_nodes(graph, step, n)
	
	# node_features_list = graph.nodes.data() # list
	# single_node_feature_tuple = node_features_list[0] #tuple
	# single_node_feature_list = single_node_feature_tuple[1].keys() # second element of the tuple is dict whichj represents the node features, we want feature list
	
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

def positive_class_node_feat(graph, n, random = False):
	'''Kreira dataframe koji ce da sadrzi feature pozitivne klase.
		Bira se nasumicnih n veza

		n - definise koliko ce biti redova u samom dataframe-u
	'''

	features_list = list(list(graph.nodes.data())[0][1].keys())
	df = pd.DataFrame(columns = features_list.insert(0, "CLASS"), index = None)
	features_list.remove("CLASS")
	rand_edges = []
	rand_indexes = []

	if random:
		while n != 0:
			rand = randint(0, len(list(graph.edges)) - 1)
			if rand in rand_indexes:
				continue
			else:
				rand_edges.append(list(graph.edges)[rand])
				rand_indexes.append(rand)
				n -= 1
	else:
		rand_edges = list(graph.edges)[:n]
		
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

def negative_class_edge_feat(graph, step, n, add_weight = False):
	'''Kreira dataframe koji ce da sadrzi feature negativne klase.
	Featuri se kreiraju pomocu funkcija iz metric_functions modula

	n - definise koliko ce biti redova u samom dataframe-u
	'''

	node_pairs = get_rand_close_nodes(graph, step, n)
	first_edge = list(graph.edges())[0]
	features_list = list(graph.get_edge_data(first_edge[0], first_edge[1]).keys())
	df = pd.DataFrame(columns = features_list.insert(0, "CLASS"), index = None)
	features_list.remove("CLASS")

	for pair in node_pairs:
		try:
			pair = tuple(pair)
			row = {pair: {"CLASS": 0}}
			
			row[pair]["CN"] = common_neighbors(graph, pair[0], pair[1])
			row[pair]["JC"] = round(jaccards_coefficient(graph, pair[0], pair[1]), 4)
			row[pair]["PA"] = preferential_attachment(graph, pair[0], pair[1])
			row[pair]["AA"] = round(adamic_adar(graph, pair[0], pair[1]), 4)
			row[pair]["RA"] = round(resource_allocation(graph, pair[0], pair[1]), 4)

			if add_weight:
				row = add_weight_to_edges(graph, pair, row)
			
			df = df.append(row[pair], ignore_index = True)
		except KeyError:
			print("Dogodio se KeyError!")
			continue

	return df

def positive_class_edge_feat(graph, n, random = False):
	'''Kreira dataframe koji ce da sadrzi feature pozitivne klase.
	Featuri se kreiraju pomocu funkcija iz metric_functions modula.
	Bira se nasumicnih n veza

	n - definise koliko ce biti redova u samom dataframe-u
	'''

	first_edge = list(graph.edges())[0]
	features_list = list(graph.get_edge_data(first_edge[0], first_edge[1]).keys())
	df = pd.DataFrame(columns = features_list.insert(0, "CLASS"), index = None)
	features_list.remove("CLASS")
	rand_edges = []
	rand_indexes = []

	if random:
		while n != 0:
			rand = randint(0, len(list(graph.edges)) - 1)
			if rand in rand_indexes:
				continue
			else:
				rand_edges.append(list(graph.edges)[rand])
				rand_indexes.append(rand)
				n -= 1
	else:
		rand_edges = list(graph.edges)[:n]

	for edge in rand_edges:
		try:
			row = {"CLASS": 1}

			row.update(graph.get_edge_data(edge[0], edge[1]))
			
			df = df.append(row, ignore_index = True)
		except KeyError:
			print("Dogodio se KeyError!")
			continue
	
	return df

# -------------------------------------------------------

def extract_feat_by_feat(graph, feat_path, featnames_path):
	'''Izvlači feature po feature i tako ih upisuje
	u graf (ne grupise ih)
	'''

	featnames =[]
	with open(featnames_path, "r") as featnames_file:
		for line in featnames_file:
			line = line.strip("\n").split(" ", 3)
			featnames.append("F" + line[3])
	
	with open(feat_path, "r") as dataset:
		dic = {}
		for line in dataset:
			dic.clear()
			features = line.strip("\n").split(" ")
			node = features[0]
			dic = {node : {}}
			del features[0]
			for i in range(len(features)):
				dic[node][featnames[i]] = features[i]
			nx.set_node_attributes(graph, dic)

	return graph

def group_features(graph, featnames_path, feat_path):
	'''Grupise feature noda na osnovu .featnames fajla
	'''

	# Ovde se upisuju featuri u dictionary. Ime featurea je key,
	# a value je max vrednost koju feature moze da ima,
	# skala ide od 0 - max vrednost
	name_holder = " "
	max_val  = 0
	features = {}
	with open(featnames_path, "r") as featnames:
		for line in featnames:
			line = line.strip("\n").split(" ")
			if name_holder != line[1]:
				features[name_holder] = max_val
				name_holder = line[1]
				max_val  = 0
			max_val += 1

		del features[" "]

	# Ovde se koricenjem prethodnog dict featuri grupisu i upisuju u
	# kao atributi noda
	dic = {}
	with open(feat_path, "r") as feat:
		for line in feat:
			dic.clear()
			line = line.strip("\n").split(" ")
			node = line[0]
			dic = {node: {}}
			del line[0]

			for feat_name, max_val in features.items():
				for index, item in enumerate(line[:max_val]):
					if item == "1":
						dic[node][feat_name] = index
						break
					if index == max_val-1:
						dic[node][feat_name] = "-1"
				del line[:max_val]

			# Ovde se menjaju nazivi kijeva, jer ako se ne promene
			# ne mogu se upisati u .gml fajl
			i = 0
			keys = list(dic[node].keys())
			for old_key in keys:
				dic[node]["F" + str(i)] = dic[node][old_key]
				del dic[node][old_key]
				i += 1

			nx.set_node_attributes(graph, dic)
		
		return graph

def calculate_edge_features(graph, add_weight = False):
	'''Racuna feature za svaki edge i upisuje ih u graf.
		Koristi implementirane metrike iz metric_functions modula.
		Ako je add_weight True onda se pomocu featurea nodova 
		racuna tezina veza i upisuje u dataframe 
	'''

	dic = {}
	for edge in graph.edges():
		dic.clear()
		dic = {edge: {}}
		dic[edge]["CN"] = common_neighbors(graph, edge[0], edge[1])
		dic[edge]["JC"] = round(jaccards_coefficient(graph, edge[0], edge[1]), 4)
		dic[edge]["PA"] = preferential_attachment(graph, edge[0], edge[1])
		dic[edge]["AA"] = round(adamic_adar(graph, edge[0], edge[1]), 4)
		dic[edge]["RA"] = round(resource_allocation(graph, edge[0], edge[1]), 4)

		if add_weight:
			dic = add_weight_to_edges(graph, edge, dic)

		nx.set_edge_attributes(graph, dic)

	return graph

def add_weight_to_edges(graph, edge, dic):
	'''Daje tezinu vezama izmedju nodova na osnovu featura nodova.
	Poredi feature po feature od nodova i ukupan broj poklapanja upisuje
	kao tezina veze
	'''

	num = 0
	node_features_list = list(list(graph.nodes.data())[0][1].keys())
	for key in node_features_list:
		try:
			if graph.nodes[edge[0]][key] == graph.nodes[edge[1]][key]:
				num += 1
		except KeyError:
			print("Dogodio se KeyError! \n")
			continue

	dic[edge]["WGH"] = num

	return dic

# -------------------------------------------------------

def write_all_featnames_in_file(ego_indexes, combined_featnames_path):
	'''Spaja sve feat u jedan .featnames fajl. Ego_indexes su indeksi svih fajlova
		kroz koje iterira. Koristi indekse featurea i tako ih upisuje u fajl.
	'''

	ls = []
	for index in ego_indexes:
		featnames_file = os.path.join(dirname, 
			"Raw_datasets\\ego-facebook\\{}.featnames".format(index))
		with open(featnames_file, "r") as featnames:
			for line in featnames:
				line = line.split(" ", 3)
				feat = "F" + line[3]
				if feat not in ls:
					ls.append(feat)
	
	
	with open(combined_featnames_path, "w") as combined_featnames:
		for elem in ls:
			combined_featnames.write(elem)

def fill_other_features(graph, combined_featnames_path):
	'''Prolazi sve kroz feat nodova i tamo gde nedostaje feat
		upisuje ga i zadaje mu inicijalnu vrednost 0.
	'''

	featnames = []
	with open(combined_featnames_path, "r") as featnames_file:
		for line in featnames_file:
			line = line.strip("\n")
			featnames.append(line)

	a = len(featnames)
	dic = {}
	for data in graph.nodes.data():
		dic.clear()
		node = data[0]
		attributes = data[1]
		dic = {node: attributes}

		for i in range(len(featnames)):
			if not featnames[i] in dic[node].keys():
				dic[node][featnames[i]] = "0"

		nx.set_node_attributes(graph, dic) 

	return graph		
