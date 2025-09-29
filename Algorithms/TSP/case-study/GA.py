# Generative Algorithm for Travelling Salesman Problem #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Wed 19 Mar 2025 @ 20:05:01 +0100
# Modified: Sat 22 Mar 2025 @ 00:07:46 +0100

# Packages
import numpy as np
import random
from itertools import permutations
from Graph import *

class GeneticAlgorithm(object):
    '''
    A class to solve the Travelling Salesman Problem (TSP) using a Genetic Algorithm.

    Attributes:
        adj_matrix (ndarray):   A symmetric matrix representing distances between
                                cities.
        pop_size (int):         Number of individuals in the population.
        generations (int):      Number of maximum iterations of the algorithm.
        mutation_rate (float):  Probability of mutation per individual.
        elitism (bool):         Whether to retain the best solution in each
                                generation.
    '''

    def __init__(self, adj_matrix, pop_size=100, generations=100, 
                 mutation_rate=0.1, elitism=True, initial_population=None):
        self.adj_matrix = adj_matrix
        self.num_cities = len(self.adj_matrix)
        self.population_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.elitism = elitism
        self.population = initial_population if initial_population else self.initialize_pop()

    def initialize_pop(self):
        '''
        Generate an initial population of random routes.
        '''
        return [random.sample(range(self.num_cities), self.num_cities) for _ 
                in range(self.population_size)]

    def fitness(self, route):
        '''
        Computes fitness as the inverse of route distance
        '''
        return 1 / self.route_distance(route)

    def route_distance(self, route):
        '''
        Calculates the total distance of a given route.
        '''
        return sum(self.adj_matrix[route[i]][route[i+1]] for i in 
                   range(len(route) - 1)) + self.adj_matrix[route[-1]][route[0]]

    def select_parent(self):
        '''
        Selects a parent using tournament selection.
        '''
        tournament_size = 5
        tournament = random.sample(self.population, tournament_size)
        return min(tournament, key = self.route_distance)

    def ordered_crossover(self, parent1, parent2):
        '''
        Performs ordered crossover on two parents.
        '''
        start, end = sorted(random.sample(range(self.num_cities), 2))
        child = [None] * self.num_cities
        child[start:end] = parent1[start:end]
        remaining = [city for city in parent2 if city not in child]
        for i in range(self.num_cities):
            if child[i] is None:
                child[i] = remaining.pop(0)
        return child

    def mutate(self, route):
        '''
        Performs swap mutation with a given probability.
        '''
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(self.num_cities), 2)
            route[i], route[j] = route[j], route[i]
        return route

    def evolve_population(self):
        '''
        Evolves the population to the next generation.
        '''
        new_population = []
        if self.elitism:
            new_population.append(min(self.population, key=self.route_distance))
        while len(new_population) < self.population_size:
            parent1, parent2 =  self.select_parent(), self.select_parent()
            child = self.ordered_crossover(parent1, parent2)
            new_population.append(self.mutate(child))
        self.population = new_population

    def run(self, verbose=False):
        '''
        Run the Genetic Algorithm and returns the best found route and distance.
        '''
        for gen_number in range(self.generations):
            if verbose:
                print(f"Generation {gen_number+1}")
            else:
                pass
            self.evolve_population()
        best_route = min(self.population, key = self.route_distance)
        best_route_closed = best_route + [best_route[0]]
        return best_route_closed, self.route_distance(best_route) 
