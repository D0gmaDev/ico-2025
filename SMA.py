import random

from mesa import Agent, Model
from mesa.datacollection import DataCollector

from ico import *
from load_database import load_data

from RS import RS

POOL_SIZE = 10

class RSAgent(Agent):

    def __init__(self, model):
        super().__init__(model)
        self.solution = None
        self.fitness = None

    def step(self):
        print(f"start agent {self.unique_id}")
        self.start_solution = self.model.solution_pool[random.randint(0, len(self.model.solution_pool) - 1)]
        self.start_fitness = fitness(self.model.state, self.start_solution, self.model.distance_matrix)

        self.solution, self.fitness = RS(self.model.state, self.start_solution, self.model.distance_matrix, iterations=30)

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

        self.datacollector = DataCollector(
            agent_reporters={
                "solution": lambda a: a.solution,
                "fitness" : lambda a: a.fitness
                }
        )

        for _ in range(3):
            RSAgent(self)

    def step(self):
        self.datacollector.collect(self)
        self.agents.do("step")
        self.agents.do("advance")

        self.solution_pool.sort(key=lambda x: fitness(self.state, x, self.distance_matrix))
        self.solution_pool = self.solution_pool[:POOL_SIZE]
        self.best_solution = self.solution_pool[0]


def construct_initial_solution(state):
    """
    Génère une solution initiale en visitant les clients dans l'ordre, avec des retours au dépôt
    lorsque la capacité du camion est atteinte.
    """
    orders = state["orders"]
    solution = [0]  # Commence au dépôt
    current_load = 0

    for client_id in range(1, len(orders)):
        if current_load + orders[client_id] > q:  # Si la capacité est dépassée
            solution.append(0)  # Retour au dépôt
            current_load = 0  # Réinitialiser la charge
        solution.append(client_id)
        current_load += orders[client_id]

    solution.append(0)  # Retour final au dépôt
    return solution


state = load_data()
distance_matrix = compute_distance_matrix(state)
solution = construct_initial_solution(state)

model = VRPModel(state, distance_matrix, [solution] * POOL_SIZE)

for i in range(10):
    print(f"Step {i}")
    model.step()

print(model.best_solution)
print(fitness(state, model.best_solution, distance_matrix))