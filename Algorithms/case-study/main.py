# Main Experiment Script #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Sat 22 Mar 2025 @ 16:13:18 +0100
# Modified: Mon 24 Mar 2025 @ 17:47:39 +0100

# Packages
import numpy as np
import time
import csv
import os
import pandas
from tabulate import tabulate
import argparse
from Graph import *
from NN import *
from GA import *
from GPM import *
from SA import *
from ACO import *

# Create a Graph class instance
def generate_graph(size, min_dist=10, max_dist=1000):
    '''
    Function which generates a Graph instance.

    Args:
        size (int):     Number of nodes (cities) the graph has.
        min_dist (int): Minimum distance between 2 cities.
        max_dist (int): Maximum distance between 2 cities.

    Returns:
        object:         A weighted graph data structure of the desired 
                        specifications.
    '''
    G = graph_generator(size, min_dist, max_dist)
    graph = next(G)
    return graph

# Initialize the population for the Genetic Algorithms
def greedy_pop(pop_size, graph):
    '''
    Function that generates the initial population of the Genetic Algorithm by 
    using the Greedy Permuting Method introduced by Junjun Liu and Wenzheng Li.

    Args:
        pop_size (int):     Population size (agents) of the Genetic Algorithm.
        graph (object):     The graph to be studied for an optimal tour.

    Returns:
        greedy_pop (list):  A list of different agents in the population, each
                            comprising of different genes.
    '''
    initializer = GreedyPermutingTSP(graph.adj_matrix)
    pop = initializer.generate_population(pop_size)
    return pop

def Nearest_Neighbor_Case_Study(graph):
    '''
    Function that calls the Nearest Neighbor algorithm and greedily solves the
    Travelling Salesman Problem.

    Args:
        graph (object): The graph to be studied for an optimal tour.

    Returns:
        list:   A list of graph vertices that comprise the optimal tour.
        float:  The total distance of the optimal tour.
    '''
    route, distance = NN_statistics(graph)
    return route, distance

def Genetic_Algorithm_Case_Study(graph, pop_size=650, gens=200, mut_rate=0.1, 
        elitism=True, init_pop=None):
    '''
    Function that calls the default Genetic Algorithm, which tries to solve the
    Travelling Salesman Problem without implementing any trick at all.

    Args:
        graph (object):     The graph to be studied for an optimal tour.
        pop_size (int):     Population size (agents) of the Genetic Algorithm.
        gens (int):         Number of generations to evolve using Genetics.
        mut_rate (float):   The rate at which the generations mutate.
        elitism (bool):     Implement meritocracy to the population (keep only
                            the best candidates in the gene pool, and evolve 
                            from there).
        init_pop (list):    Initialize the population of the algorithm in a 
                            custom way (set to None to avoid errors).

    Returns:
        list:   A list of graph vertices that comprise the optimal tour.
        float:  The total distance of the optimal tour.
    '''
    # Create an instance of the Genetic Algorithm solver.
    GA = GeneticAlgorithm(graph.adj_matrix, pop_size=pop_size, generations=gens,
                        mutation_rate=mut_rate, elitism=elitism, 
                        initial_population=init_pop)
    # Call the solver on the initialized instance.
    route, distance = GA.run(verbose=False)
    return route, distance

def Simulated_Annealing_Case_Study(graph, init_temp=1000, cool_rate=0.995, 
        stop_temp=1e-3):
    '''
    Function that calls the default Simulated Annealing Algorithm, which tries
    to solve the Travelling Salesman Problem without implementing any trick.

    Args:
        graph (object):     The graph to be studied for an optimal tour.
        init_temp (int):    Initial temperature of the annealing process.
        cool_rate (float):  Cooling rate of the annealing process.
        stop_temp (int):    Stopping criterion of the annealing process. Similar
                            to tolerance in iterative algorithms.

    Returns:
        list:   A list of graph vertices that comprise the optimal tour.
        float:  The total distance of the optimal tour.
    '''
    # Create an instance of the Simulated Annealing solver.
    SA = SimulatedAnnealing(graph.adj_matrix, initial_temp=init_temp,
                            cooling_rate=cool_rate, stopping_temp=stop_temp)
    # Call the solver on the initialized instance.
    route, distance = SA.run()
    return route, distance

def Ant_Colony_Case_Study(graph, ants=20, merit=5, iterations=100, decay=0.5,
        alpha=1, beta=2):
    '''
    Function that calls the Ant Colony Optimization Algorithm, which solves the
    Travelling Salesman Problem by simulating the functions of an ant colony
    searching for food.

    Args:
        graph (object): The graph to be studied for an optimal tour.
        ants (int):     Number of ants in the colony.
        merit (int):    The amount of paths to remember in the iterations and
                        reinforce using pheromones.
        decay (float):  The rate at which the ant pheromones decay.
        alpha (int):    Parameter to influence the importance of pheromone
                        trails.
        beta (int):     Parameter to influence the importance of the heuristic
                        information, i.e., how much ants like a path (edge).

    Returns:
        list:   A list of graph vertices that comprise the optimal tour.
        float:  The total distance of the optimal tour.
    '''
    # Create an instance of the Ant Colony solver.
    ACO = AntColony(graph.adj_matrix, n_ants=ants, n_best=merit, n_iterations=
            iterations, decay=decay, alpha=alpha, beta=beta)
    # Call the solver on the initialized instance.
    route, distance = ACO.run()
    return route, distance

def baseline(size, method=None, RUNS=30):
    '''
    Function which calls all of the heuristic algorithms that solve the 
    Travelling Salesman Problem, runs them using their default settings, and
    establishes a baseline comparison by saving the output of each solver in a
    CSV file.

    Args:
        size (int):         Number of nodes (cities) a graph has.
        method (string):    The name of the method to be tracked.
        RUNS (int):         The number of random case studies to be performed
                            to keep track of the changes in the data.
    '''
    results = []
    timings = []
    methods = ["NN", "GA", "GPM", "SA", "ACO"]

    # Raise ValueError if the method argument is not within the list of accepted
    # algorithmic solvers.
    if method not in methods:
        raise ValueError("Unrecognised method, see documentation for valid seceltion.")
    
    # Loop which runs each algorithm 30 (default) times and records the results
    # in a .csv files for later.
    for run in range(1, RUNS + 1):
        print(f"{method}: Run {run}/{RUNS}")
        graph = generate_graph(size=size)
        if method == "NN":
            start = time.time()
            _, dist = Nearest_Neighbor_Case_Study(graph)
            end = time.time()
            
            elapsed = end - start
            results.append(dist)
            timings.append(elapsed)
            filename = "BaselineNearestNeighbor.csv"
        elif method == "GA":
            start = time.time()
            _, dist = Genetic_Algorithm_Case_Study(graph)
            end = time.time()

            elapsed = end - start
            results.append(dist)
            timings.append(elapsed)
            filename = "BaselineGeneticAlgorithm.csv"
        elif method == "GPM":
            pop_size = 100
            GPM_pop = greedy_pop(pop_size, graph)
            start = time.time()
            _, dist = Genetic_Algorithm_Case_Study(graph, init_pop = GPM_pop)
            end = time.time()

            elapsed = end - start
            results.append(dist)
            timings.append(elapsed)
            filename = "GreedyGeneticAlgorithm.csv"
        elif method == "SA":
            start = time.time()
            _, dist = Simulated_Annealing_Case_Study(graph)
            end = time.time()

            elapsed = end - start
            results.append(dist)
            timings.append(elapsed)
            filename = "BaselineSimulatedAnnealing.csv"
        elif method == "ACO":
            start = time.time()
            _, dist = Ant_Colony_Case_Study(graph)
            end = time.time()

            elapsed = end - start
            results.append(dist)
            timings.append(elapsed)
            filename = "BaselineAntColony.csv"
        else:
            raise ValueError("Error in the loop. Requires manual debugging.")

    try:
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Run", "Distance", "Time (s)"])

            for i in range(RUNS):
                writer.writerow([i + 1, results[i], timings[i]])
    finally:
        file.close()

def tracker(size, method=None, RUNS=30):
    '''
    Function that calls the fine-tuned version of a specific solver and saves 
    its performance output (distance of the optimal tour and uptime) to a CSV
    file.

    Args:
        size (int):         Number of nodes (cities) a graph has.
        method (string):    The name of the method to be tracked.
        RUNS (int):         The number of random case studies to be performed
                            to keep track of the changes in the data.
    '''
    results = []
    timings = []
    methods = ["NN", "GA", "GPM", "SA", "ACO"]
    
    # Raise ValueError if the method argument is not within the list of accepted
    # algorithmic solvers.
    if method not in methods:
        raise ValueError("Unrecognised method, see documentation for valid seceltion.")
    
    # Loop which runs each algorithm 30 (default) times and records the results
    # in a .csv files for later.
    for run in range(1, RUNS + 1):
        print(f"{method}: Run {run}/{RUNS}")
        graph = generate_graph(size=size)
        if method == "NN":
            start = time.time()
            _, dist = Nearest_Neighbor_Case_Study(graph)
            end = time.time()
            
            elapsed = end - start
            results.append(dist)
            timings.append(elapsed)
            filename = "NearestNeighbor.csv"
        elif method == "GA":
            pop_size = 650
            gens = 300
            mut_rate = 0.1
            start = time.time()
            _, dist = Genetic_Algorithm_Case_Study(graph, pop_size, gens, mut_rate)
            end = time.time()

            elapsed = end - start
            results.append(dist)
            timings.append(elapsed)
            filename = "GeneticAlgorithm.csv"
        elif method == "GPM":
            pop_size = 100
            GPM_pop = greedy_pop(pop_size, graph)
            start = time.time()
            _, dist = Genetic_Algorithm_Case_Study(graph, init_pop = GPM_pop)
            end = time.time()

            elapsed = end - start
            results.append(dist)
            timings.append(elapsed)
            filename = "GreedyGeneticAlgorithm.csv"
        elif method == "SA":
            init_temp = 5000
            cool_rate = 0.9999
            stop_temp = 1e-8
            start = time.time()
            _, dist = Simulated_Annealing_Case_Study(graph, init_temp, cool_rate,
                    stop_temp)
            end = time.time()

            elapsed = end - start
            results.append(dist)
            timings.append(elapsed)
            filename = "SimulatedAnnealing.csv"
        elif method == "ACO":
            ants = 40
            start = time.time()
            _, dist = Ant_Colony_Case_Study(graph, ants)
            end = time.time()

            elapsed = end - start
            results.append(dist)
            timings.append(elapsed)
            filename = "AntColony.csv"
        else:
            raise ValueError("Error in the loop. Requires manual debugging.")

    try:
        with open(filename, mode="w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["Run", "Distance", "Time (s)"])

            for i in range(RUNS):
                writer.writerow([i + 1, results[i], timings[i]])
    finally:
        file.close()

def main(generate_baseline=False):
    parser = argparse.ArgumentParser(description="Run optimization algorithms.")
    parser.add_argument("algorithm", choices=["NN", "GA", "GPM", "SA", "ACO"], help="Choose the algorithm to run.")

    args = parser.parse_args()
    if args.algorithm == "NN":
        tracker(229, method="NN", RUNS=30)
        print("Nearest Neighbor case study complete...")
    elif args.algorithm == "GA":
        if generate_baseline:
            baseline(229, method="GA", RUNS=30)
            print("Baseline for Genetic Algorithm established...")
        tracker(229, method="GA", RUNS=30)
        print("Genetic Algorithm case study complete...")
    elif args.algorithm == "GPM":
        tracker(229, method="GPM", RUNS=30)
        print("Greedy Genetic Algorithm case study complete...")
    elif args.algorithm == "SA":
        if generate_baseline:
            baseline(229, method="SA", RUNS=30)
            print("Baseline for Simulated Annealing established...")
        tracker(229, method="SA", RUNS=30)
        print("Simulated Annealing case study complete...")
    elif args.algorithm == "ACO":
        baseline(229, method="ACO", RUNS=30)
        print("Baseline for Ant Colony established...")
        tracker(229, method="ACO", RUNS=30)
        print("Ant Colony case study complete...")

if __name__ == "__main__":
    generate_baseline=False
    main(generate_baseline)
