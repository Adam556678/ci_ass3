from genetic_algorithm import GeneticAlgorithm
import random
import time
import math
import numpy as np
from typing import List

class OneMaxGA(GeneticAlgorithm):
    """
    Genetic algorithm for solving the One-Max problem.
    Inherits from the GeneticAlgorithm abstract base class.
    """

    def __init__(self, population_size: int, chromosome_length: int, crossover_prob:float, mutation_rate: float, elitism_num: int, r_min:int, r_max:int, tournament_size : int, mean :int, std : int):
        """
        Initialize the OneMaxGA instance.

        Args:
            population_size (int): Size of the population.
            chromosome_length (int): Length of each chromosome (bitstring).
            mutation_rate (float): Probability of mutation for each bit.
        """
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.crossover_prob = crossover_prob
        self.mutation_rate = mutation_rate
        self.elitism_num = elitism_num
        self.r_min = r_min
        self.r_max = r_max
        self.tournament_size = tournament_size
        self.mean = mean
        self.std = std
        self.population = self.initialize_population()
       
    def create_individual(self) -> List[float]:
        """
        Create a new individual (random bitstring).

        Returns:
            List[int]: A newly created individual.
        """
        return [random.uniform(self.r_min,self.r_max) for _ in range(self.chromosome_length)]    
    
    def initialize_population(self) -> List[List[float]]:
        """
        Initialize the population with random bitstrings.

        Returns:
            List[List[int]]: Initial population.
        """

        return [self.create_individual() for _ in range(self.population_size)]
    
    # def standard_decoder(self, chromosome: List[int]):
    #     chromosome_length = len(chromosome)
        
    #     sum = 0
    #     for i in range(chromosome_length):
    #         sum += chromosome[i] * (2**(chromosome_length-i-1))
        
    #     chromosome_real_val = self.min + (sum / 2 ** chromosome_length) * (self.max - self.min)
    #     return math.ceil(chromosome_real_val)
                
    # def gray_decoder(self, chromosome: List[int]) -> int:
    #     chromosome_length = len(chromosome)
    #     total_sum = 0

    #     for i in range(chromosome_length):
    #         xk_sum = 0
    #         for j in range(i + 1):
    #             xk_sum += chromosome[j]
    #         total_sum += (xk_sum % 2) * (2 ** (chromosome_length - i - 1))

    #     chromosome_val = self.min + ((total_sum / (2 ** chromosome_length)) * (self.max - self.min))
    #     return math.ceil(chromosome_val)
        
    def evaluate_fitness(self, chromosome: List[float]) -> float:
        fitness = 8 - ((chromosome[0] + 0.0317) ** 2) + (chromosome[1] ** 2)
        return fitness        
        
        

    # def evaluate_fitness_gray(self, chromosome: List[int]) -> int:
    #     chromosome_length = len(chromosome)
    #     mid = math.ceil(chromosome_length / 2)
    #     chromosome_x1_gray_value = self.gray_decoder(chromosome=chromosome[:mid])
    #     chromosome_x2_gray_value = self.gray_decoder(chromosome=chromosome[mid:])

    #     fitness = 8 - ((chromosome_x1_gray_value + 0.0317) ** 2) + (chromosome_x2_gray_value ** 2)
    #     return fitness

    # def linear_rank_selection(self, pop: List[List[int]], sp: float):
    #     population_size = len(pop)
    #     # population_fitness = [self.evaluate_fitness(indiv) for indiv in pop]
        
        
    #     ranks = np.array(population_fitness).argsort().argsort() + 1

    #     pop_linear_rank_fitness = [(2-sp) + 2 * (sp - 1) * (rank-1)/(population_size-1) for rank in ranks]


    #     return pop_linear_rank_fitness
        
    
    # def calculate_cumulative_probabilities(self, pop_fitness) -> List[float]:
    #     """
    #     Calculate cumulative probabilities for each individual.

    #     Returns:
    #         List[float]: Cumulative probabilities.
    #     """
    #     total_fitness = sum(fit for fit in pop_fitness)
    #     probabilities = [fit / total_fitness for fit in pop_fitness]
    #     cumulative_probabilities = [sum(probabilities[:i + 1]) for i in range(len(pop_fitness))]
    #     return cumulative_probabilities

    # def select_parents(self) -> List[float]:
    #     parents = random.choices(self.population, k=self.tournament_size)
    #     fitnesses = [self.evaluate_fitness(individual) for individual in parents]
    #     best_individual_index = fitnesses.index(max(fitnesses))
    #     return parents[best_individual_index]

    def select_parents(self,) -> List[float]:
        parents = random.sample(self.population, k=self.tournament_size)
        fitnesses = []
        
        for individual in parents:
            fitnesses.append(self.evaluate_fitness(individual))        

        best_indiv_loc = fitnesses.index(max(fitnesses))
        
        return parents[best_indiv_loc]


    def crossover(self, parent1: List[float], parent2: List[float], alpha : float) -> List[List[float]]:
        """
        Perform one-point crossover between two parents.

        Args:
            parent1 (List[int]): First parent chromosome.
            parent2 (List[int]): Second parent chromosome.

        Returns:
            List[List[int]]: Two offspring chromosomes.
        """
        
        child1 = [(alpha * p1) + ((1 - alpha) * p2) for p1, p2 in zip(parent1, parent2)]
        child2 = [(alpha * p2) + ((1 - alpha) * p1) for p1, p2 in zip(parent1, parent2)]
        return parent1, parent2
    

    def mutate(self, chromosome: List[float]) -> List[float]:
        """
        Apply bit flip mutation to an individual.

        Args:
            chromosome (List[int]): The chromosome to be mutated.

        Returns:
            List[int]: The mutated chromosome.
        """
        mutated_chromosome = chromosome.copy()
        chromosome_length = len(mutated_chromosome)

        for i in range(chromosome_length):
            rand_num = np.random.rand()
            if rand_num < self.mutation_rate:
                gaussian_rand = np.random.normal(self.mean, self.std, 1)
                mutated_chromosome[i] = mutated_chromosome[i] + gaussian_rand
        
        return mutated_chromosome     
        

    def elitism(self) -> List[List[float]]:
        """
        Apply elitism to the population (keep the best two individuals).

        Args:
            new_population (List[List[int]]): The new population after crossover and mutation.
        """
        sorted_population = sorted(self.population, key=self.evaluate_fitness, reverse=True)
        best_individuals = sorted_population[:self.elitism_num]
        return best_individuals
    

    def run(self, max_generations):
        best_solutions = []
        for generation in range(max_generations):
            new_population = []
            while len(new_population) < self.population_size:
                parent1= self.select_parents()
                parent2 = self.select_parents()
                offspring1, offspring2 = self.crossover(parent1, parent2, alpha=0.3)
                offspring1 = self.mutate(offspring1)
                offspring2 = self.mutate(offspring2)
                new_population.extend([offspring1, offspring2])

            new_population = new_population[0:self.population_size-self.elitism_num] # make sure the new_population is the same size of original population - the best individuals we will append next
            best_individuals = self.elitism()
            new_population.extend(best_individuals)
            self.population = new_population


        best_solutions.append(max(self.population, key=self.evaluate_fitness))                
        return best_solutions

if __name__ == "__main__":
    tournament_sizes = [2, 4]
    for k in tournament_sizes:
        population_size = 100
        chromosome_length = 2 
        crossover_prob = 0.6
        mutation_rate = 0.05
        elitism_num = 2
        max_generations = 150
        r_min = -2
        r_max = 2
        
        mean = 30
        std = 15

        start = time.time()
        onemax_ga = OneMaxGA(population_size, chromosome_length,crossover_prob, mutation_rate,elitism_num, r_min=r_min, r_max=r_max, tournament_size=k, mean=mean, std=std)
        best_solutions = onemax_ga.run(max_generations)
        ga_time = time.time()-start
        print("GA Solution Time:",round(ga_time,1),'Seconds')
        print(f"tournament size: {k}")
        print(f"Best solution: {best_solutions[0]}")
        print(f"Fitness: {onemax_ga.evaluate_fitness(best_solutions[0])}")
        