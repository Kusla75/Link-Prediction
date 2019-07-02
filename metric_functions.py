from math import log10

def intersection(list1, list2):
    inter_list = []
    for value in list1:
        if value in list2:
            inter_list.append(value)
    return inter_list


def common_neighbors(graph, n1, n2):
    ngh1 = list(graph.neighbors(n1))
    ngh2 = list(graph.neighbors(n2))

    if n1 in ngh2:                      # Pošto veza postoji onda su nod 1 i nod 2 međusobno povezani i
        ngh1.remove(n2)                 # moraju se ukloniti iz međusobnih lista komšija, jer merimo vrednost
        ngh2.remove(n1)                 # za međusobne veze koje ne postoje a mogle bi (pravimo se kao da veza 
                                        # između noda 1 i noda 2 ne postoji)
    return len(intersection(ngh1, ngh2))

def jaccards_coefficient(graph, n1, n2):
    score = 0
    ngh1 = list(graph.neighbors(n1))
    ngh2 = list(graph.neighbors(n2))

    if n1 in ngh2:   
        ngh1.remove(n2)                 
        ngh2.remove(n1)

    try:
        score = len(intersection(ngh1, ngh2))/len(ngh1 + ngh2)
    except ZeroDivisionError:
        score = 0

    return score


def preferential_attachment(graph, n1, n2):
    ngh1 = list(graph.neighbors(n1))
    ngh2 = list(graph.neighbors(n2))

    if n1 in ngh2:
        ngh1.remove(n2)
        ngh2.remove(n1)

    return len(ngh1) * len(ngh2)


def adamic_adar(graph, n1, n2):
    score = 0
    ngh1 = list(graph.neighbors(n1))
    ngh2 = list(graph.neighbors(n2))

    if n1 in ngh2:
        ngh1.remove(n2)
        ngh2.remove(n1)

    inter_list = intersection(ngh1, ngh2)

    for node in inter_list:
        try:
            score += 1/log10(len(list(graph.neighbors(node))))
        except ZeroDivisionError:
            continue
 
    return score


def resource_allocation(graph, n1, n2):
    score = 0
    ngh1 = list(graph.neighbors(n1))
    ngh2 = list(graph.neighbors(n2))

    if n1 in ngh2:
        ngh1.remove(n2)
        ngh2.remove(n1)

    inter_list = intersection(ngh1, ngh2)

    for node in inter_list:
        try:
            score += 1/len(list(graph.neighbors(node)))
        except ZeroDivisionError:
            continue

    return score
