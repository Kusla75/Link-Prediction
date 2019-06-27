from math import log10

def intersection(list1, list2):
    inter_list = []
    for value in list1:
        if value in list2:
            inter_list.append(value)
    return inter_list

