# Nearest Neightbour Algorithm for Travelling Salesman Problem #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 17 Mar 2025 @ 19:27:23 +0100
# Modified: Wed 19 Mar 2025 @ 23:19:37 +0100

# Packages
import numpy as np
from Graph import *

# Nearest Neightbour Algorithm
def nearest_neightbour(graph):
    '''
    Function that implements the Nearest Neightbour Algorithm
    Input: graph (Graph data type)
    '''
    # Fetch the adjacency matrix, and graph size
    distances = graph.adj_matrix
    size = graph.size

    route = [0]                     # start at city A (node 0)
    visited = set([0])              # set of visited cities
    while len(visited) < size:
        current_city = route[-1]    # get current position
#        nearest_city = min([(i, distances[current_city][i]) for i in range(size)
#                            if i not in visited and 
#                            distances[current_city][i] != 0], 
#                            key=lambda x: x[1])[0]
        nearest_city = min([(i, distances[current_city][i]) for i in range(size)
                            if i not in visited], key=lambda x: x[1])[0] 
        route.append(nearest_city)
        visited.add(nearest_city)
    route.append(0)
    return route

def NN_statistics(graph):
    route = nearest_neightbour(graph)
    distance = 0
    for i in range(len(route) - 1):
        distance += graph.fetch_weight(route[i], route[i+1])
    return route, distance
