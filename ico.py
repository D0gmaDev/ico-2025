import numpy as np

q = 30 # Truck capacity
omega = 50 # Truck cost

def fitness(state, solution, distance_matrix):
    orders = state["orders"]

    delivered = set()
    capacity = q
    total_distance = 0
    
    penalty_count = 0  # Count violations instead of returning early
    penalty_factor = 1000  # Base penalty multiplier

    trucks = 0

    for i in range(1, len(solution)):
        prev, curr = solution[i-1], solution[i]
        
        # Check if the path exists
        if distance_matrix[prev, curr] == float("inf"):
            penalty_count += 6  # Path does not exist

        total_distance += distance_matrix[prev, curr]
        
        # If we're at a customer, deliver their order
        if curr != 0:  
            if capacity < orders[curr]:
                penalty_count += (orders[curr] - capacity + 1) // 2  # Not enough capacity
            
            capacity -= orders[curr]
            delivered.add(curr)
        
        # If we're back at the depot, refill the truck
        if curr == 0:
            trucks += 1
            capacity = q

    # Ensure all orders are delivered
    if len(delivered) < len(orders) - 1:  # Excluding depot
        penalty_count += 5 * (len(orders) - 1 - len(delivered))  # Some orders were not delivered

    # Final fitness score: total distance + trucks + penalty
    
    print(trucks)

    trucks_value = trucks * omega
    penalty_value = penalty_count * penalty_factor
    return total_distance + trucks_value + penalty_value
    

def compute_distance_matrix(state):
    coords = state["position"]
    n = len(coords)

    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_matrix[i, j] = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2)
    return dist_matrix