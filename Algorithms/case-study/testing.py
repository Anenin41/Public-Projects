# Script that tests various aspects of the code #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Mon 17 Mar 2025 @ 17:48:05 +0100
# Modified: Sat 22 Mar 2025 @ 19:12:32 +0100

# Packages
import numpy as np
import time
from Graph import *
from NN import *
from GA import *
from GPM import *
from SA import *
from ACO import *

# Create a graph
def generate_instance(size, min_dist, max_dist):
    G = graph_generator(size, min_dist, max_dist)
    graph_instance = next(G)
    return graph_instance

# Initialize the population
def initialize_pop(pop_size, graph_instance):
    initializer = GreedyPermutingTSP(graph_instance.adj_matrix)
    greedy_pop = initializer.generate_population(pop_size)
    return greedy_pop

# Test Nearest Neighbor Algorithm
def test_NN(graph_instance):
    route_NN, distance_NN = NN_statistics(graph_instance)
    return route_NN, distance_NN

# Test (default) Genetic Algorithm
def test_GA(graph_instance, pop_size=100, gens=100, mut_rate=0.1, 
                elitism=True):
    # Initialize GeneticAlgorithm class (object)
    GA = GeneticAlgorithm(graph_instance.adj_matrix, pop_size=pop_size, 
                          generations=gens, mutation_rate=mut_rate,
                          elitism=elitism)
    route_GA, distance_GA = GA.run()
    return route_GA, distance_GA

# Test Genetic Algorithm with Greedy Population Initialization
def test_GPM(graph_instance, pop_size=100, gens=100, mut_rate=0.1,
             elitism=True, init_pop=None):
    # Initialize GeneticAlgorithm class (object)
    GPM = GeneticAlgorithm(graph_instance.adj_matrix, pop_size=pop_size,
                           generations=gens, mutation_rate=mut_rate,
                           elitism=elitism, initial_population=init_pop)
    route_GPM, distance_GPM = GPM.run()
    return route_GPM, distance_GPM

# Test the Simulated Annealing Algorithm
def test_SA(graph_instance, init_temp=1000, cool_rate=0.995, stop_temp=1e-3):
    # Initialize SimulatedAnnealing class (object)
    SA = SimulatedAnnealing(graph_instance.adj_matrix, initial_temp=init_temp,
                            cooling_rate=cool_rate, stopping_temp=stop_temp)
    route_SA, distance_SA = SA.run()
    return route_SA, distance_SA

# Test the Ant Colony Optimization Algorithm
def test_ACO(graph_instance, ants=20, best_ants=5, iterations=100, decay=0.2,
             alpha=1, beta=2):
    # Initialize AntColony class (object)
    ACO = AntColony(graph_instance.adj_matrix, n_ants=ants, n_best=best_ants,
                    n_iterations=iterations, decay=decay, alpha=1, beta=2)
    route_ACO, distance_ACO = ACO.run()
    return route_ACO, distance_ACO

def main():
    graph = generate_instance(229, 10, 1000)
    ants = 40
    merit = 5
    iterations = 100
    decay = 0.5
    alpha = 1
    beta = 2
    _, distance = test_ACO(graph, ants, merit, iterations, decay, alpha, beta)

    print(f"ACO: Distance: {distance}, Ants: {ants}, Best Ants: {merit}, Iterations: {iterations}, Decay: {decay}, Alpha: {alpha}, Beta: {beta}")

if __name__ == "__main__":
    main()
