from abc import ABC, abstractmethod
from typing import List

class GeneticAlgorithm(ABC):
    """
    Abstract base class for genetic algorithms.
    """

    @abstractmethod
    def create_individual(self) -> List[float]:
        """
        Create a new individual (random bitstring).

        Returns:
            List[int]: A newly created individual.00
        """
        pass

    @abstractmethod
    def initialize_population(self) -> List[List[float]]:
        """
        Initialize the population of individuals.

        Returns:
            List[List[int]]: A list of bitstrings (chromosomes).
        """
        pass
    
    # @abstractmethod
    # def standard_decoder(self, chromosome:List[float]):
    #     pass
    
    # @abstractmethod
    # def gray_decoder(self, chromosome:List[float]):
    #     pass

    @abstractmethod
    def evaluate_fitness(self, chromosome: List[float]) -> float:
        """
        Evaluate the fitness of an individual.

        Args:
            chromosome (List[int]): The bitstring representing an individual.

        Returns:
            int: Fitness value (e.g., sum of 1s in the bitstring).
        """
        pass

    # @abstractmethod
    # def linear_rank_selection(self, pop:List[List[float]], sp:float):
    #     pass

    @abstractmethod
    def select_parents(self,) -> List[List[float]]:
        """
        Select two parent chromosomes from the population.

        Returns:
            List[List[int]]: Two parent chromosomes.
        """
        pass

    @abstractmethod
    def crossover(self, parent1: List[float], parent2: List[float], alpha : float) -> List[List[float]]:
        """
        Perform crossover (recombination) between two parents.

        Args:
            parent1 (List[int]): First parent chromosome.
            parent2 (List[int]): Second parent chromosome.

        Returns:
            List[List[int]]: Two offspring chromosomes.
        """
        pass

    @abstractmethod
    def mutate(self, chromosome: List[float]) -> List[float]:
        """
        Apply mutation to an individual.

        Args:
            chromosome (List[int]): The chromosome to be mutated.

        Returns:
            List[int]: The mutated chromosome.
        """
        pass

    # @abstractmethod
    # def elitism(self) -> List[List[float]]:
    #     """
    #     Apply elitism to the population (e.g., keep the best individuals).

    #     Args:
    #         new_population (List[List[int]]): The new population after crossover and mutation.
    #     """
    #     pass
    
