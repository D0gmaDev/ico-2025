import random

from ico import *

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


def tabu_search(state, initial_solution, distance_matrix, iterations=500, tabu_tenure=7, neighborhood_size=25):
    best_solution = initial_solution
    best_cost = fitness(state, best_solution, distance_matrix)
    tabu_list = []

    q = state["q"]
    
    for _ in range(iterations):
        neighborhood = []

        # Générer un sous-ensemble de voisins avec des swaps
        for _ in range(neighborhood_size):
            i, j = random.sample(range(1, len(best_solution) - 1), 2)  # Choisir 2 clients aléatoires
            new_solution = best_solution[:]
            new_solution[i], new_solution[j] = new_solution[j], new_solution[i]
            
            if is_valid_solution(new_solution, state['orders'], q):
                cost = fitness(state, new_solution, distance_matrix)
                move = (i, j)
                neighborhood.append((new_solution, cost, move))
        
        if not neighborhood:
            print("Aucun voisin valide trouvé, arrêt de la recherche.")
            break

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
