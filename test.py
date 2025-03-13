from ico import *

# Sample state with positions and orders
state = {
    "position": [
        (0, 0),  # Depot (0)
        (5, 2), (3, -3), (2, 2), (1, 0),  # Customers (1-4)
        (3, 3), (2, 1), (-5, 0), (7, 1),  # Customers (5-8)
        (-3, -2), (5, 1), (1, 2), (4, 2)  # Customers (9-12)
    ],
    "orders": [0, 5, 10, 7, 8, 6, 9, 4, 3, 1, 2, 3, 7]  # Order demand per customer (0 is depot)
}

# Compute the distance matrix
distance_matrix = compute_distance_matrix(state)

# Initial solution
solution = [0, 9, 7, 0, 2, 0, 4, 6, 10, 8, 1, 12, 5, 3, 11, 0]  # Returns to depot

# Compute the fitness of the initial solution
print(fitness(state, solution, distance_matrix))
plot_solution(state, solution)