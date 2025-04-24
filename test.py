from ico import *

from RS import RS
from rs_c_wrapper import rs_c_optimize

from Tabou import tabu_search
from tabu_c_wrapper import tabu_c_optimize

import time

# Sample state with positions and orders
state = {
    "position": [
        (0, 0),  # Depot (0)
        (5, 2), (3, -3), (2, 2), (1, 0),  # Customers (1-4)
        (3, 3), (2, 1), (-5, 0), (7, 1),  # Customers (5-8)
        (-3, -2), (5, 1), (1, 2), (4, 2)  # Customers (9-12)
    ],
    "orders": [0, 5, 10, 7, 8, 6, 9, 4, 3, 1, 2, 3, 7],  # Order demand per customer (0 is depot)
    "q": 400,  # truck capacity
    "omega": 10  # truck cost
}

# Compute the distance matrix
distance_matrix = compute_distance_matrix(state)

# Initial solution
initial_solution = [0, 9, 7, 0, 2, 0, 4, 6, 10, 8, 1, 12, 5, 3, 11, 0]  # Returns to depot

# Parameters for both algorithms
iterations = 200
T = 1000
T_min = 0.1
alpha = 0.99

print("Initial solution fitness:", fitness(state, initial_solution, distance_matrix))
print("\nRunning Python RS implementation...")
start_time = time.time()
python_solution, python_fitness = RS(state, initial_solution, distance_matrix, iterations, T, T_min, alpha)
python_time = time.time() - start_time

print("\nRunning C RS implementation...")
start_time = time.time()
c_solution, c_fitness = rs_c_optimize(state, initial_solution, distance_matrix, iterations, T, T_min, alpha)
c_time = time.time() - start_time

print("\nResults:")
print("-" * 50)
print(f"Python RS:")
print(f"  Time: {python_time:.3f} seconds")
print(f"  Final fitness: {python_fitness:.2f}")
print(f"  Solution: {python_solution}")
print(f"\nC RS:")
print(f"  Time: {c_time:.3f} seconds")
print(f"  Final fitness: {c_fitness:.2f}")
print(f"  Solution: {c_solution}")
print(f"\nSpeedup: {python_time/c_time:.2f}x")

# Tabou
print("\n")

iterations = 20000

print("Initial solution fitness:", fitness(state, initial_solution, distance_matrix))
print("\nRunning Python Tabou implementation...")
start_time = time.time()
python_solution, python_fitness = tabu_search(state, initial_solution, distance_matrix, iterations)
python_time = time.time() - start_time

print("\nRunning C Tabou implementation...")
start_time = time.time()
c_solution, c_fitness = tabu_c_optimize(state, initial_solution, distance_matrix, iterations)
c_time = time.time() - start_time

print("\nResults:")
print("-" * 50)
print(f"Python Tabou:")
print(f"  Time: {python_time:.3f} seconds")
print(f"  Final fitness: {python_fitness:.2f}")
print(f"  Solution: {python_solution}")
print(f"\nC Tabou:")
print(f"  Time: {c_time:.3f} seconds")
print(f"  Final fitness: {c_fitness:.2f}")
print(f"  Solution: {c_solution}")
print(f"\nSpeedup: {python_time/c_time:.2f}x")
