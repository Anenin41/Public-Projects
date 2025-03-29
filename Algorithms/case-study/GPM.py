# The Greedy Permuting Generic Algorithm for the Travelling Salesman Problem #
# Author: Konstantine Garas
# E-mail: kgaras041@gmail.com // k.gkaras@student.rug.nl
# Created: Wed 19 Mar 2025 @ 21:06:54 +0100
# Modified: Wed 19 Mar 2025 @ 22:29:48 +0100

# Packages
from GA import *

class GreedyPermutingTSP(GeneticAlgorithm):
    '''
    A child of the GeneticAlgorithm class that implements the greedy permuting
    method.
    '''
    def __init__(self, adj_matrix):
        super().__init__(adj_matrix)

    def get_nearest_neighbors(self, city_index, neighbor_number):
        '''
        Finds the nearest neighbors of a city.
        '''
        distances = [(i, self.adj_matrix[city_index][i]) for i in 
                     range(self.num_cities) if i != city_index]
        distances.sort(key=lambda x: x[1])
        return [city for city, _ in distances[:neighbor_number]]

    def greedy_permuting(self, cities):
        '''
        Greedy permutation of remaining cities based on nearest neighbors.
        '''
        if not cities:
            return []
        current_city = cities[0]
        tour = [current_city]
        unvisited = set(cities) - {current_city}
        while unvisited:
            next_city = min(unvisited, key=lambda city: 
                            self.adj_matrix[current_city][city])
            tour.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
        return tour

    def generate_population(self, population_size, neighbor_number=5):
        '''
        Generate an initial population using the greedy method.
        '''
        population = []
        city_index = 0
        while city_index < self.num_cities:
            individual = [city_index]
            neighbors = self.get_nearest_neighbors(city_index, neighbor_number)
            for neighbor in neighbors:
                individual.append(neighbor)
                rest_cities = self.greedy_permuting([city for city in 
                                                     range(self.num_cities) if
                                                     city not in individual])
                individual.extend(rest_cities)
                population.append(individual)
            city_index += 1
        return sorted(population, key=lambda route: self.route_distance(route))[:population_size]
