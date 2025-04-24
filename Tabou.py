import random

from ico import *

def tabu_search(state, initial_solution, distance_matrix, iterations=500, tabu_tenure=7, neighborhood_size=25):
    best_solution = initial_solution
    best_cost = fitness(state, best_solution, distance_matrix)
    tabu_list = []
    
    for _ in range(iterations):
        neighborhood = []

        # Générer un sous-ensemble de voisins avec des swaps
        for _ in range(neighborhood_size):
            i, j = random.sample(range(1, len(best_solution) - 1), 2)  # Choisir 2 clients aléatoires
            new_solution = best_solution[:]
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            
            cost = fitness(state, new_solution, distance_matrix)
            move = (i, j)
            neighborhood.append((new_solution, cost, move))
        
        if not neighborhood:
            # print("Aucun voisin valide trouvé, arrêt de la recherche.")
            continue

        # Trier les voisins par coût
        neighborhood.sort(key=lambda x: x[1])

        for candidate, cost, move in neighborhood:
            if move not in tabu_list or cost < best_cost:
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
