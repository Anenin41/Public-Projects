# Ant Colony Optimization Algorithm for Travelling Salesman Problem #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Fri 21 Mar 2025 @ 13:17:18 +0100
# Modified: Fri 21 Mar 2025 @ 14:10:07 +0100

# Packages
import numpy as np
import random

class AntColony(object):
    '''
    Ant Colony Optimization algorithm for solving the Travelling Salesman Problem

    Attributes:
        adj_matrix (ndarray): Symmetric adjacency matrix representing distances
        n_ants (int): Number of ants to simulate per iterations
        n_best (int): Number of top-performing ants to reinforce pheromones.
        n_iterations (int): Number of generations to simulate.
        decay (float): Pheromone evaporation rate (0 < decay < 1).
        alpha (float): Influence of pheromone trail strength.
        beta (float): Influence of heuristic desirability (1 / distance [common])
        pheromone (ndarray): Matrix tracking pheromone levels for each edge
        num_cities (int): Total number of cities (nodes).
    '''
    def __init__(self, adj_matrix, n_ants=20, n_best=5, n_iterations=100, 
                 decay=0.5, alpha=1, beta=2):
        self.adj_matrix = adj_matrix
        # Initialize all edges with equal pheromone number
        self.pheromone = np.ones((len(adj_matrix), len(adj_matrix)))
        self.all_inds = range(len(adj_matrix))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.num_cities = len(adj_matrix)

    def pick_next(self, current, visited):
        '''
        Chooses the next city to visit using the ACO probability rule.

        Args:
            current (int): Current city index.
            visited (set): Set of already visited cities.

        Returns:
            int: Next city index.
        '''
        pheromone = np.copy(self.pheromone[current])
        pheromone[list(visited)] = 0        # Avoid revisiting cities
        
        distances = np.copy(self.adj_matrix[current])
        distances[list(visited)] = 10**6   # Avoid zero division
        
        desirability = (1.0 / distances) ** self.beta
        probabilities = (pheromone ** self.alpha) * desirability
        total = np.sum(probabilities)

        if total == 0:
            return random.choise([i for i in self.all_inds if i not in visited])

        probabilities /= total
        return np.random.choice(self.all_inds, 1, p=probabilities)[0]

    def spread_pheromone(self, all_paths, n_best):
        '''
        Updates pheromone levels based on the top-performing ant tours.

        Args:
            all_paths (list): List of tuples (path, distance) for all ants.
            n_best (int): Number of top paths to use for reinforcement.
        '''
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, distance in sorted_paths[:n_best]:
            for move in zip(path, path[1:] + [path[0]]):    # Include return
                self.pheromone[move] += 1.0 / distance      # to start

    def path_distance(self, path):
        '''
        Calculates the total distance of a path (includes return to start).

        Args:
            path (list): A list of city indices.

        Returns:
            float: Total tour distance.
        '''
        return sum(self.adj_matrix[path[i]][path[i+1]] for i in 
                   range(len(path) - 1)) + self.adj_matrix[path[-1]][path[0]]

    def gen_path(self, start):
        '''
        Construct a complete path for one ant using probabilistic selection.

        Args:
            start (int): Starting city index.

        Returns:
            list: A complete tour of the graph.
        '''
        path = [start]
        visited = set(path)

        for _ in range(self.num_cities - 1):
            next_city = self.pick_next(path[-1], visited)
            path.append(next_city)
            visited.add(next_city)

        return path

    def gen_all_paths(self):
        '''
        Generate complete paths for all ants in the current iteration.

        Returns:
            list: List of tuples [(path, distance), ...] for each ant.
        '''
        all_paths = []
        for _ in range(self.n_ants):
            path = self.gen_path(start=0)
            distance = self.path_distance(path)
            all_paths.append((path, distance))
        return all_paths

    def run(self, verbose=False):
        '''
        Executes the ACO algorithm and returns the best tour and its distance.

        Returns:
            tuple: (best_route, best_distance)
                best_route (list) = List of city indices representing the cities.
                best_distance (float) = Total distance of the best tour.
        '''
        shortest_path = None
        all_time_shortest_path = ([], np.inf)

        for iteration in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            shortest_path = min(all_paths, key=lambda x: x[1])
            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.spread_pheromone(all_paths, self.n_best)
            self.pheromone *= self.decay    # Evaporate pheromones

        if verbose:
            print(f"Iteration {iteration+1}: shortest path length = {shortest_path[1]:.f}")

        # Close the tour by returning to the starting city
        best_route_closed = all_time_shortest_path[0] + [
                all_time_shortest_path[0][0]]
        return best_route_closed, all_time_shortest_path[1]
