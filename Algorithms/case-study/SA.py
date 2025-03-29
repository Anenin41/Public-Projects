# Simulated Annealing Algorithm for Travelling Salesman Problem #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Wed 19 Mar 2025 @ 22:39:36 +0100
# Modified: Fri 21 Mar 2025 @ 12:56:19 +0100

# Packages
import numpy as np
import random

class SimulatedAnnealing(object):
    '''
    A class to solve the Traveling Salesman Problem using Simulated Annealing.

    Attributes:
        adj_matrix (ndarray):   A symmetric matrix representing distances 
                                between cities.
        initial_temp(float):    Starting temperature for the annealing process.
        cooling_rate(float):    Rate at which temperature decreases.
        stopping_temp(float):   Threshold at which the algorithm stops.
    '''

    def __init__(self, adj_matrix, initial_temp=1000, cooling_rate=0.995,
                 stopping_temp=1e-3):
        self.adj_matrix = adj_matrix
        self.num_cities = len(adj_matrix)
        self.initial_temp = initial_temp
        self.cooling_rate = cooling_rate
        self.stopping_temp = stopping_temp
        self.current_solution = self.initial_solution()
        self.best_solution = list(self.current_solution)
        self.current_distance = self.route_distance(self.current_solution)
        self.best_distance = self.current_distance

    def initial_solution(self):
        '''
        Generates an initial random tour.
        '''
        solution = list(range(self.num_cities))
        random.shuffle(solution)
        return solution

    def route_distance(self, route):
        '''
        Calculates the total distance of a given route.
        '''
        return sum(self.adj_matrix[route[i]][route[i+1]] for i in 
                   range(len(route) - 1)) + self.adj_matrix[route[-1]][route[0]]

    def swap_cities(self, route):
        '''
        Generates a new neighbor solution by swapping two cities.
        '''
        new_route = list(route)
        i, j = random.sample(range(self.num_cities), 2)
        new_route[i], new_route[j] = new_route[j], new_route[i]
        return new_route

    def accept_solution(self, new_distance, current_distance, temp):
        '''
        Determines whether to accept a new solution based on probability.
        '''
        if new_distance < current_distance:
            return True
        return random.random() < np.exp((current_distance - new_distance) / temp)

    def run(self):
        '''
        Executes the Simulated Annealing Algorithm.
        '''
        temp = self.initial_temp
        while temp > self.stopping_temp:
            new_solution = self.swap_cities(self.current_solution)
            new_distance = self.route_distance(new_solution)
            if self.accept_solution(new_distance, self.current_distance, temp):
                self.current_solution = new_solution
                self.current_distance = new_distance
                if new_distance < self.best_distance:
                    self.best_solution = new_solution
                    self.best_distance = new_distance
            temp *= self.cooling_rate
        best_route_closed = self.best_solution + [self.best_solution[0]]
        return best_route_closed, self.best_distance
