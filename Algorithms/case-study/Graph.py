# Graph Data Structure - Class Implementation #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 17 Mar 2025 @ 17:20:22 +0100
# Modified: Mon 24 Mar 2025 @ 00:42:23 +0100

# Packages
import numpy as np
import random
#import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    # Initialize the Graph
    def __init__(self, size):
        '''
        Initialize a snapshop of a graph using adjacency matrix.
        '''
        # Adjacency Matrix
        self.adj_matrix = [[0] * size for _ in range(size)]
        # Graph Size
        self.size = size
        # Names of the Vertices
        self.vertex_data = [''] * size

    # Overwrite size
    def overwrite_size(self, new_size):
        '''
        Forcefully overwrite the size value of the graph.
        Input: new_size (Int)
        '''
        self.size = new_size

    # Overwrite Vertices
    def overwrite_vertex_data(self, new_vertex_data:list):
        '''
        Forcefully overwrite the Vertex set of the graph.
        Input: new_vertex_data (List)
        '''
        if type(new_vertex_data) != list:
            raise TypeError("Vertices must be stored into a list")
        else:
            self.vertex_data = new_vertex_data

    # Re-initialize Vertices
    def re_init_vertices(self):
        '''
        Re-initialize the Vertex set of the graph.
        '''
        self.vertex_data = [''] * self.size

    # Assume an undirected but weighted graph
    def add_edge(self, u, v, weight):
        '''
        Manually add an Edge to the graph.
        Input:
            u: Node u
            v: Node v
            weight: weight of the edge
        '''
        if 0 <= u < self.size and 0 <= v < self.size:
            self.adj_matrix[u][v] = weight
            self.adj_matrix[v][u] = weight  # apply symmetry

    # Fetch the weight of a specific edge (simple implementation due to 
    # summetric property of the graph)
    def fetch_weight(self, u, v):
        '''
        Fetch the weight of a specific edge from the adjacency matrix.
        Input:
            u: Node u
            v: Node v
        '''
        if 0 <= u < self.size and 0 <= v < self.size:
            return self.adj_matrix[u][v]

    # Completely overwrite Adjacency matrix with a new entry
    # Note: this function desyncs the vertices
    def overwrite_adj_matrix(self, matrix):
        '''
        Forcefully overwrite the adjacency matrix of the graph.
        Input: matrix (symmetric otherwise ValueError
        '''
        if np.shape(matrix)[0] != np.shape(matrix)[1]:
            raise ValueError("New adjacency matrix is not square.")
        else:
            self.adj_matrix = matrix                    # update adjacency matrix
            self.overwrite_size(np.shape(matrix)[0])    # update graph size
            self.re_init_vertices()                     # re-init vertices

    # Add the names of the Vertex
    def add_vertex_data(self, vertex, data):
        '''
        Manually add a Vertex in the set.
        Input:
            vertex: position in the Vertex set
            data: name, value, or some other data type
        '''
        if 0 <= vertex < self.size:
            self.vertex_data[vertex] = data
    
    # Print the Adjacency Matrix
    def print_adj_matrix(self):
        '''
        Print the adjacency matrix.
        '''
        print("\nAdjacency Matrix:")
        for row in self.adj_matrix:
            print(" ".join(f"{val:4}" for val in row))

    # Print the Vertices
    def print_vertex_data(self):
        '''
        Print the Vertex set of the graph.
        '''
        print("\nVertex Data:")
        for vertex, data in enumerate(self.vertex_data):
            print(f"Vertex {vertex}: {data}")

    # Print both the Adjacency Matrix and the Vertices
    def print_graph(self):
        '''
        Print both the adjacency matrix and the vertex set of the graph.
        '''
        self.print_adj_matrix()
        self.print_vertex_data()

# Generate a complete Graph of random distances for a given number of cities
def graph_generator(size, min_distance=1, max_distance=100):
    ''' 
    Generate a graph with a random adjacency matrix.
    Inputs:
        size: graph size.
        min_distance: minimum integer value to pick from.
        max_distance: maximum integer value to pick from.
    '''
    while True:
        # Create an instance of the Graph class
        graph = Graph(size)
        # Generate a random symmetric adjacency matrix
        for i in range(size):
            for j in range(i + 1, size):
                # Only upper triangular block to ensure symmetry
                weight = random.randint(min_distance, max_distance)
                graph.add_edge(i, j, weight)
    
        # Yield the graph instead of returning it
        yield graph

# Plot the Graph by using the Adjacency Matrix and NetworkX
#def plot_graph(graph):
#    '''
#    Graph plotter using NetworkX and Matplotlib
#    Input: graph (Graph data type)
#    '''
#    G = nx.Graph()
#
#    # Add nodes with labels
#    for i in range(graph.size):
#        node_label = graph.vertex_data[i] if graph.vertex_data[i] else f"V{i}"
#        G.add_node(i, label = node_label)
#
#    # Add edges with weights
#    for i in range(graph.size):
#        for j in range(i + 1, graph.size):  # traverse symmetry structrue
#            if graph.adj_matrix[i][j] > 0:
#                G.add_edge(i, j, weight = graph.adj_matrix[i][j])
#
#    # Position the nodes
#    pos = nx.spring_layout(G, seed=42)
#    plt.figure(figsize=(8,6))
#    nx.draw(G, pos, with_labels=True, labels=nx.get_node_attributes(G, "label"),
#            node_color="lightblue", edge_color="gray", node_size=1500, 
#            font_size=10)
#
#    # Draw edge labels (weights)
#    edge_labels = {(i, j): graph.adj_matrix[i][j] for i, j in G.edges()}
#    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=10)
#
#    plt.title("Graph Visualization")
#    plt.show()
