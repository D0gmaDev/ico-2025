import numpy as np
import matplotlib.pyplot as plt

q = 400 # Truck capacity
omega = 10 # Truck cost

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

    # Ensure the path starts and ends at the depot
    if solution[0] != 0 or solution[-1] != 0:
        penalty_count += 2  # Route does not start/end at depot

    # Final fitness score: total distance + trucks + penalty
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

def plot_solution(state, solution):
    plt.figure(figsize=(8, 8))

    # Extraire les coordonnées
    pos = state["position"]

    # Tracer les points
    for i, (x, y) in enumerate(pos):
        if i == 0:
            plt.scatter(x, y, color='black', s=100)  # Point 0 en noir
        else:
            plt.scatter(x, y, color='white', edgecolors='black', s=100)
            #plt.text(x, y, str(i), fontsize=6, color='red')

    # Tracer les lignes pour chaque véhicule
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow']

    # Split solutions by vehicle
    vehicles = []
    current_vehicle = [0]

    for node in solution:
        current_vehicle.append(node)
        if node == 0:
            vehicles.append(current_vehicle)
            current_vehicle = [0]
            
    if current_vehicle:
        vehicles.append(current_vehicle)

    for i, route in enumerate(vehicles):
        for j in range(len(route) - 1):
            (x1, y1), (x2, y2) = pos[route[j]], pos[route[j + 1]]
            plt.plot([x1, x2], [y1, y2], color=colors[i % len(colors)])

    # Affichage du titre et du graphique
    plt.title("Solution : " + ", ".join(map(str, solution)))
    plt.axis('off')  # Masquer les axes
    plt.show()
