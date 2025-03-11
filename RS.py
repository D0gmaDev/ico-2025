import numpy as np
import random
import matplotlib.pyplot as plt
import networkx as nx

    
coords = {
    0: (250, 250),
    1: (490, 315),  2: (277, 196), 3: (141, 385), 4: (301, 355),
    5: (111, 304), 6: (69, 140), 7: (418, 213), 8: (281, 79),
    9: (361, 129), 10: (347, 273), 11: (400, 340), 12: (300, 300)}

routes = {
    "Véhicule 1": [0, 1, 2, 0],  
    "Véhicule 2": [0, 3, 4, 5, 6, 0], 
    "Véhicule 3": [0, 7, 8, 9, 10, 0],  
    "Véhicule 4": [0, 11, 12, 0]  
}

def distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def total_distance(routes):
    total_dist = 0  # Initialisation de total_dist à 0
    for route in routes.values():
        for i in range(len(route) - 1):
            total_dist += distance(coords[route[i]], coords[route[i + 1]])
        total_dist += distance(coords[route[-1]], coords[route[0]])  # Retour au point de départ
    return total_dist

def mutate(routes):
    vehicle_1, vehicle_2 = random.sample(list(routes.keys()), 2)
    
    # Choisir deux points au hasard dans ces véhicules
    if len(routes[vehicle_1]) > 1 and len(routes[vehicle_2]) > 1:
        idx_1 = random.randint(1, len(routes[vehicle_1]) - 2)
        idx_2 = random.randint(1, len(routes[vehicle_2]) - 2)
        
        # Effectuer l'échange
        routes[vehicle_1][idx_1], routes[vehicle_2][idx_2] = routes[vehicle_2][idx_2], routes[vehicle_1][idx_1]
    return routes

def RS(routes):
    # Initialisation
    current_routes = routes.copy()
    current_distance = total_distance(current_routes)
    
    best_routes = current_routes.copy()
    best_distance = current_distance
    
    # Paramètres du recuit simulé
    T = 1000  # Température initiale
    T_min = 0.1  # Température minimale
    alpha = 0.99  # Facteur de refroidissement
    iterations = 1000  # Nombre d'itérations
    
    # Exécution du recuit simulé
    while T > T_min:
        for _ in range(iterations):
            new_routes = mutate(current_routes.copy())
            new_distance = total_distance(new_routes)
            
            # Si la nouvelle solution est meilleure, l'accepter
            if new_distance < current_distance:
                current_routes = new_routes
                current_distance = new_distance
                # Mettre à jour la meilleure solution
                if current_distance < best_distance:
                    best_routes = current_routes.copy()
                    best_distance = current_distance
            # Sinon, l'accepter avec une certaine probabilité
            else:
                delta = new_distance - current_distance
                probability = np.exp(-delta / T)
                if random.random() < probability:
                    current_routes = new_routes
                    current_distance = new_distance
        
        T *= alpha  # Réduire la température

    return best_routes, best_distance

best_routes, best_distance = RS(routes)
print("Meilleur chemin:", best_routes)  # Correction ici
print("Meilleure distance:", best_distance)


def plot(coords, best_routes):
    """
    Affiche le meilleur chemin trouvé pour plusieurs véhicules.
    
    :param coords: Dictionnaire des coordonnées des points
    :param best_routes: Dictionnaire représentant les routes de chaque véhicule
    """
    plt.figure(figsize=(8, 8))

    # Tracer les points
    for point in coords:
        if point == 0:
            plt.scatter(coords[point][0], coords[point][1], color='black', s=100)  # Point 0 en noir
        else:
            plt.scatter(coords[point][0], coords[point][1], color='white', edgecolors='black', s=100)
            plt.text(coords[point][0] + 5, coords[point][1] + 5, str(point), fontsize=12, color='black')

    # Tracer les lignes pour chaque véhicule
    colors = ['blue', 'green', 'red', 'purple']
    for i, (vehicle, route) in enumerate(best_routes.items()):
        for j in range(len(route) - 1):
            p1, p2 = route[j], route[j + 1]
            plt.plot([coords[p1][0], coords[p2][0]], [coords[p1][1], coords[p2][1]], color=colors[i])
        # Relier le dernier point au premier
        plt.plot([coords[route[-1]][0], coords[route[0]][0]], 
                 [coords[route[-1]][1], coords[route[0]][1]], color=colors[i])

    # Affichage du titre et du graphique
    plt.title("Recuit Simulé - Meilleurs Chemins (Multiples Véhicules)")
    plt.axis('off')  # Masquer les axes
    plt.show()

plot(coords, best_routes)
