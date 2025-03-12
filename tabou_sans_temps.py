#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:02:33 2025

@author: eliottdaniel
"""

import matplotlib.pyplot as plt
from ico import compute_distance_matrix, fitness,q,omega

# Recharger le module ico pour prendre en compte les modifications


# Afficher les valeurs de q et omega pour vérification
print(f"Valeur de q (capacité du camion) : {q}")
print(f"Valeur de omega (coût par camion) : {omega}")

def construct_initial_solution():
    """ Retourne la solution initiale fournie sous forme de liste plate """
    return [0, 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 10, 11, 12, 0]

def count_trucks(solution):
    """ Compte le nombre de camions utilisés (nombre de retours au dépôt) """
    return solution.count(0) - 1  # On retire le premier 0 (départ)

def is_valid_solution(solution, orders, capacity):
    """ Vérifie que chaque camion respecte sa capacité """
    total_load = 0
    for node in solution:
        if node == 0:  # Retour au dépôt -> Réinitialisation de la charge
            total_load = 0
        else:
            total_load += orders[node]
            if total_load > capacity:
                return False  # Dépassement de la capacité
    return True

def tabu_search(state, initial_solution, max_iter=100, tabu_tenure=10):
    distance_matrix = compute_distance_matrix(state)
    best_solution = initial_solution
    best_cost = fitness(state, best_solution, distance_matrix)
    tabu_list = []
    
    for iteration in range(max_iter):
        neighborhood = []

        # Générer des voisins avec des swaps
        for i in range(1, len(best_solution) - 1):  # Ignorer le dépôt (0)
            for j in range(1, len(best_solution) - 1):
                if i != j:
                    new_solution = best_solution[:]
                    new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
                    
                    if is_valid_solution(new_solution, state['orders'], q):  # Utiliser q ici
                        cost = fitness(state, new_solution, distance_matrix)
                        move = (i, j)
                        neighborhood.append((new_solution, cost, move))
        
        if not neighborhood:
            print("Aucun voisin valide trouvé, arrêt de la recherche.")
            break  # Si aucun voisin n'est valide, on arrête

        # Trier les voisins par coût
        neighborhood.sort(key=lambda x: x[1])

        for candidate, cost, move in neighborhood:
            if move not in tabu_list or cost < best_cost:  # Permet d'ignorer la liste tabou si on trouve mieux
                best_solution = candidate
                best_cost = cost
                tabu_list.append(move)
                
                if len(tabu_list) > tabu_tenure:
                    tabu_list.pop(0)

                if cost < best_cost:
                    best_cost = cost
                    tabu_list = []  # Réinitialisation de la liste tabou si une meilleure solution est trouvée
                break
    
    return best_solution, best_cost

def plot_solution(state, solution):
    positions = state["position"]
    depot = positions[0]
    clients = positions[1:]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(*depot, c='red', marker='s', label='Dépôt')
    for i, (x, y) in enumerate(clients, start=1):
        plt.scatter(x, y, c='blue')
        plt.text(x, y, f"{i}\n(q={state['orders'][i]})", fontsize=10, verticalalignment='bottom', horizontalalignment='right')
    
    colors = ['g', 'b', 'm', 'c', 'y', 'orange', 'purple']
    truck_count = 0
    current_color = colors[truck_count % len(colors)]
    
    for i in range(len(solution) - 1):
        start, end = solution[i], solution[i + 1]
        if start == 0:  # Nouveau camion
            truck_count += 1
            current_color = colors[truck_count % len(colors)]
        plt.plot([positions[start][0], positions[end][0]], [positions[start][1], positions[end][1]], color=current_color, linewidth=2, marker='o')
    
    plt.legend()
    plt.title(f"Optimisation des tournées de véhicules - Nombre de camions : {truck_count}")
    plt.show()

# Exécution
state = {
    "position": [
        (0, 0),
        (5, 2), (3, -3), (2, 2), (1, 0),
        (3, 3), (2, 1), (-5, 0), (7, 1),
        (-3, -2), (5, 1), (1, 2), (4, 2)
    ],
    "orders": [0, 5, 10, 7, 8, 6, 9, 4, 3, 1, 2, 3, 7]
}

distance_matrix = compute_distance_matrix(state)
initial_solution = construct_initial_solution()

# Calcul du coût de la solution initiale
initial_cost = fitness(state, initial_solution, distance_matrix)
print("Coût de la solution initiale:", initial_cost)

# Optimisation avec Tabu Search
optimized_solution, min_cost = tabu_search(state, initial_solution)
nbr_trucks = count_trucks(optimized_solution)

print("Solution initiale:", initial_solution)
print("Solution optimisée:", optimized_solution)
print("Nombre de camions utilisés:", nbr_trucks)
print("Coût optimisé:", min_cost)
plot_solution(state, construct_initial_solution())