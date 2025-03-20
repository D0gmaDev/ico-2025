import random

from mesa import Agent, Model
from mesa.datacollection import DataCollector

from ico import *
from load_database import load_data

from RS import RS
from AG import AG
from Tabou import tabu_search

POOL_SIZE = 20

class RSAgent(Agent):

    def __init__(self, model):
        super().__init__(model)
        self.solution = None
        self.fitness = None

    def step(self):
        print(f"| Start RS_agent (#{self.unique_id})")
        self.start_solution = self.model.get_random_solution()
        self.start_fitness = fitness(self.model.state, self.start_solution, self.model.distance_matrix)

        self.solution, self.fitness = RS(self.model.state, self.start_solution, self.model.distance_matrix, iterations=100)

    def advance(self):
        if self.fitness <= self.start_fitness:
            self.model.solution_pool.append(self.solution)

class AGAgent(Agent):

    def __init__(self, model):
        super().__init__(model)
        self.solution = None
        self.fitness = None

    def step(self):
        print(f"| Start AG_agent (#{self.unique_id})")
        self.start_solution = self.model.get_random_solution()
        self.start_fitness = fitness(self.model.state, self.start_solution, self.model.distance_matrix)

        self.solution, self.fitness = AG(self.model.state, self.model.solution_pool[:10], self.model.distance_matrix, iterations=200, population_size=120, mutation_rate=0.2, elitism=10)

    def advance(self):
        if self.fitness <= self.start_fitness:
            self.model.solution_pool.append(self.solution)

class TabouAgent(Agent):

    def __init__(self, model):
        super().__init__(model)
        self.solution = None
        self.fitness = None

    def step(self):
        print(f"| Start Tabou_agent (#{self.unique_id})")
        self.start_solution = self.model.get_random_solution()
        self.start_fitness = fitness(self.model.state, self.start_solution, self.model.distance_matrix)

        self.solution, self.fitness = tabu_search(self.model.state, self.start_solution, self.model.distance_matrix, iterations=500)

    def advance(self):
        if self.fitness <= self.start_fitness:
            self.model.solution_pool.append(self.solution)

class VRPModel(Model):

    def __init__(self, state, distance_matrix, initial_solutions):
        super().__init__()
        self.state = state
        self.distance_matrix = distance_matrix
        self.solution_pool = initial_solutions.copy()
        self.solution_pool.sort(key=lambda x: fitness(state, x, distance_matrix))
        self.update_pool_weight()

        self.datacollector = DataCollector(
            agent_reporters={
                "solution": lambda a: a.solution,
                "fitness" : lambda a: a.fitness
                }
        )

        # Création des agents
        RSAgent(self)
        AGAgent(self)
        TabouAgent(self)

    def step(self):
        self.datacollector.collect(self)
        self.agents.do("step")
        self.agents.do("advance")

        self.solution_pool.sort(key=lambda x: fitness(self.state, x, self.distance_matrix))
        self.solution_pool = self.solution_pool[:POOL_SIZE]
        self.best_solution = self.solution_pool[0]
        print(f"-- Current fitness : {fitness(self.state, self.best_solution, self.distance_matrix)}")

    def update_pool_weight(self):
        N = len(self.solution_pool)

        # Génération d'une distribution exponentielle des probabilités
        weights = [0.9**i for i in range(N)]  # 0.9 peut être ajusté pour un effet plus ou moins fort
        total_weight = sum(weights)

        # Normalisation des poids pour en faire une distribution de probabilité
        probabilities = [w / total_weight for w in weights]
        self.pool_weights = probabilities
    
    def get_random_solution(self):
        return random.choices(self.solution_pool, weights=self.pool_weights, k=1)[0]

    def get_random_solutions(self, n):
        return random.choices(self.solution_pool, weights=self.pool_weights, k=n)


state = load_data()
distance_matrix = compute_distance_matrix(state)
solutions = construct_initial_solutions(state, POOL_SIZE)

model = VRPModel(state, distance_matrix, solutions)

steps = 10

for i in range(steps):
    print(f"[Step {i+1}/{steps}]")
    model.step()

print(model.best_solution)
print(fitness(state, model.best_solution, distance_matrix))
