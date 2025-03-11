#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:02:33 2025

@author: eliottdaniel
"""
import numpy as np
import random
import matplotlib.pyplot as plt

# Paramètres du problème
num_clients = 12
num_vehicles = 3
capacity = 38  # Capacité réduite pour tester les retours au dépôt

# Génération aléatoire des coordonnées (x, y) des clients et du dépôt
depot = (50, 50)
clients = {i: (random.randint(0, 100), random.randint(0, 100), random.randint(5, 20)) for i in range(1, num_clients + 1)}

# Création de la matrice des distances avec (0, 0) inclus
nodes = {0: depot, **clients}
distance_matrix = {
    (i, j): np.sqrt((nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)
    for i in nodes for j in nodes
}
# Ajouter une distance de 0 pour (0, 0)
distance_matrix[(0, 0)] = 0

# Fonction pour vérifier si une route respecte la capacité
def is_route_valid(route):
    total_load = sum(clients[client][2] for client in route if client != 0)
    return total_load <= capacity

# Construction initiale d'une solution avec retours au dépôt et respect de la capacité
def construct_initial_solution():
    clients_list = list(clients.keys())
    random.shuffle(clients_list)
    routes = [[] for _ in range(num_vehicles)]
    capacities = [0] * num_vehicles
    
    for client in clients_list:
        assigned = False
        for v in range(num_vehicles):
            if capacities[v] + clients[client][2] <= capacity:
                routes[v].append(client)
                capacities[v] += clients[client][2]
                assigned = True
                break
        if not assigned:
            # Retour au dépôt pour recharger
            for v in range(num_vehicles):
                if capacities[v] == 0:  # Camion vide
                    routes[v].append(0)  # Retour au dépôt
                    routes[v].append(client)
                    capacities[v] = clients[client][2]
                    assigned = True
                    break
        if not assigned:
            # Si aucun camion ne peut livrer le client, on force un retour au dépôt
            routes[0].append(0)  # Retour au dépôt
            routes[0].append(client)
            capacities[0] = clients[client][2]
    
    # Ajouter un retour au dépôt à la fin de chaque route
    for v in range(num_vehicles):
        if routes[v]:
            routes[v].append(0)
    
    # S'assurer que chaque route commence par le dépôt
    for v in range(num_vehicles):
        if routes[v] and routes[v][0] != 0:
            routes[v].insert(0, 0)  # Ajouter le dépôt au début de la route
    
    # Vérifier que toutes les routes respectent la capacité
    for route in routes:
        if not is_route_valid(route):
            print("Erreur : une route dépasse la capacité maximale.")
    
    return routes

# Fonction d'évaluation : distance totale parcourue
def total_distance(routes):
    total = 0
    for route in routes:
        for i in range(len(route) - 1):
            start, end = route[i], route[i + 1]
            if (start, end) in distance_matrix:
                total += distance_matrix[(start, end)]
            else:
                # Si la paire n'existe pas, on suppose une distance de 0 (cas de (0, 0))
                total += 0
    return total

# Algorithme de recherche tabou amélioré
def tabu_search(routes, max_iter=100, tabu_tenure=10):
    best_solution = routes
    best_cost = total_distance(routes)
    tabu_list = []
    
    for iteration in range(max_iter):
        neighborhood = []
        
        # Génération du voisinage avec des swaps intra-route et inter-routes
        for r1 in range(len(routes)):
            for i in range(1, len(routes[r1]) - 1):
                # Swaps intra-route
                for j in range(i + 1, len(routes[r1]) - 1):
                    new_route = routes[r1][:]
                    new_route[i], new_route[j] = new_route[j], new_route[i]
                    if is_route_valid(new_route):  # Vérifier la capacité
                        new_routes = routes[:r1] + [new_route] + routes[r1+1:]
                        move = (r1, i, r1, j)  # Stocker le mouvement (route1, index1, route1, index2)
                        neighborhood.append((new_routes, total_distance(new_routes), move))
                
                # Swaps inter-routes
                for r2 in range(len(routes)):
                    if r1 != r2:
                        for j in range(1, len(routes[r2]) - 1):
                            new_route1 = routes[r1][:]
                            new_route2 = routes[r2][:]
                            new_route1[i], new_route2[j] = new_route2[j], new_route1[i]
                            if is_route_valid(new_route1) and is_route_valid(new_route2):  # Vérifier la capacité
                                new_routes = routes[:r1] + [new_route1] + routes[r1+1:r2] + [new_route2] + routes[r2+1:]
                                move = (r1, i, r2, j)  # Stocker le mouvement (route1, index1, route2, index2)
                                neighborhood.append((new_routes, total_distance(new_routes), move))
        
        # Tri du voisinage par coût
        neighborhood.sort(key=lambda x: x[1])
        
        # Sélection de la meilleure solution non tabou
        for candidate, cost, move in neighborhood:
            if move not in tabu_list:
                routes = candidate
                tabu_list.append(move)
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)
                if cost < best_cost:
                    best_solution = candidate
                    best_cost = cost
                    tabu_list = []  # Réinitialiser la liste tabou si une meilleure solution est trouvée
                break
        
    
    # Vérifier que toutes les routes respectent la capacité
    for route in best_solution:
        if not is_route_valid(route):
            print("Erreur : une route dépasse la capacité maximale.")
    
    return best_solution, best_cost

# Exécution de l'algorithme
initial_routes = construct_initial_solution()
print("Solution initiale:", initial_routes)
print("Distance totale initiale:", total_distance(initial_routes))

tabu_routes, min_cost = tabu_search(initial_routes)
print("Solution optimisée:", tabu_routes)
print("Distance totale optimisée:", min_cost)

# Vérification que tous les clients sont bien livrés
delivered_clients = set(client for route in tabu_routes for client in route if client != 0)
missing_clients = set(clients.keys()) - delivered_clients
if missing_clients:
    print("Attention, les clients suivants ne sont pas livrés:", missing_clients)
else:
    print("Tous les clients sont bien livrés.")

# Visualisation graphique avec indication du sens de trajet et de la demande
def plot_solution(routes):
    plt.figure(figsize=(10, 6))
    plt.scatter(*depot, c='red', marker='s', label='Dépôt')
    for i, (x, y, q) in clients.items():
        plt.scatter(x, y, c='blue', label='Client' if i == 1 else "")
        plt.text(x, y, f"{i}\n(q={q})", fontsize=10, verticalalignment='bottom', horizontalalignment='right')
    
    colors = ['g', 'b', 'm', 'c', 'y']
    for idx, route in enumerate(routes):
        color = colors[idx % len(colors)]
        print(f"Route {idx + 1}: {route}")  # Log pour déboguer
        for i in range(len(route) - 1):
            start, end = route[i], route[i + 1]
            start_coords = nodes[start]
            end_coords = nodes[end]
            # Tracer la ligne entre les points
            plt.plot([start_coords[0], end_coords[0]], [start_coords[1], end_coords[1]], color=color, linewidth=2, marker='o', markersize=5)
            # Ajouter une flèche pour indiquer le sens de trajet
            plt.arrow(start_coords[0], start_coords[1], (end_coords[0] - start_coords[0]) * 0.9, (end_coords[1] - start_coords[1]) * 0.9, 
                      head_width=2, head_length=2, fc=color, ec=color)
    
    plt.legend()
    plt.title(f"Optimisation des tournées de véhicules\nDistance totale: {min_cost:.2f}")
    plt.show()

plot_solution(tabu_routes)