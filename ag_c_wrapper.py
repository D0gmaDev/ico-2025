import ctypes
import numpy as np
from itertools import chain

from ico import construct_initial_solutions

lib = ctypes.CDLL("./ag.cpython-313-x86_64-linux-gnu.so")

# --- Constantes (doivent correspondre à ag.h) ---
MAX_SOLUTION_LENGTH = 256
MAX_POPULATION_SIZE = 150

# --- Définir les types d'arguments et de retour ---
try:
    ag_optimize_func = lib.ag_c_optimize
    ag_optimize_func.argtypes = [
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_double,
        ctypes.POINTER(ctypes.c_double), ctypes.c_int,
        ctypes.c_int, ctypes.c_double, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
    ]
    ag_optimize_func.restype = ctypes.c_double
except AttributeError:
    print(f"Erreur: La fonction 'ag_c_optimize' n'a pas été trouvée.")
    exit(1)

# --- Fonction Wrapper Python (Modifiée) ---
def ag_c_optimize(state, start_population, distance_matrix, iterations=200, population_size=120, mutation_rate=0.2, elitism=10):
    actual_population_size = len(start_population)
    num_clients = len(state["orders"]) -1 # Nécessaire pour la génération aléatoire

    # Gérer le cas où la population initiale est trop petite
    if actual_population_size < population_size:
        num_missing = population_size - actual_population_size

        # Créer une copie pour ne pas modifier l'original en dehors de la fonction
        # Si la modification de la liste originale est acceptable, on peut utiliser:
        # current_start_population = start_population
        current_start_population = list(start_population) # Travailler sur une copie

        # Générer les solutions manquantes
        current_start_population += construct_initial_solutions(state, num_missing)

        # Remplacer la population de départ par celle complétée
        start_population = current_start_population
        # Mettre à jour la taille réelle pour les étapes suivantes
        actual_population_size = len(start_population) # Doit être égal à population_size maintenant

    elif actual_population_size > population_size:
        # Tronquer la population initiale pour correspondre à population_size
        start_population = start_population[:population_size]
        actual_population_size = population_size

    # Vérifier que la taille finale ne dépasse pas la limite C
    if actual_population_size > MAX_POPULATION_SIZE:
         raise ValueError(f"La taille finale de la population ({actual_population_size}) ne peut pas dépasser MAX_POPULATION_SIZE ({MAX_POPULATION_SIZE}) défini dans le code C.")

    # --- Préparation des arguments pour C (la suite reste similaire) ---

    # 1. Population initiale et longueurs (maintenant de taille population_size)
    solution_lengths = [len(sol) for sol in start_population]
    for i, length in enumerate(solution_lengths):
        if length > MAX_SOLUTION_LENGTH:
            raise ValueError(f"La solution à l'index {i} (potentiellement générée aléatoirement ou initiale) "
                             f"a une longueur ({length}) qui dépasse MAX_SOLUTION_LENGTH ({MAX_SOLUTION_LENGTH}) défini dans le code C.")

    flat_initial_population = list(chain.from_iterable(start_population))
    # S'assurer que la taille correspond bien à population_size *après* l'ajustement
    c_initial_population = (ctypes.c_int * len(flat_initial_population))(*flat_initial_population)
    c_solution_lengths = (ctypes.c_int * actual_population_size)(*solution_lengths) # Utiliser la taille ajustée

    # 2. Paramètres depuis 'state'
    orders = state["orders"]
    # num_clients déjà calculé plus haut
    c_orders = (ctypes.c_int * len(state["orders"]))(*state["orders"])
    c_q = ctypes.c_int(state["q"])
    c_omega = ctypes.c_double(state["omega"])

    # 3. Matrice des distances
    if not isinstance(distance_matrix, np.ndarray):
        distance_matrix = np.array(distance_matrix)
    if distance_matrix.ndim != 2 or distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("distance_matrix doit être une matrice carrée NumPy.")
    matrix_size = distance_matrix.shape[0]
    flat_dm = distance_matrix.flatten()
    c_distance_matrix = (ctypes.c_double * (matrix_size * matrix_size))(*flat_dm)
    c_matrix_size = ctypes.c_int(matrix_size)

    # 4. Paramètres de l'AG
    c_generations = ctypes.c_int(iterations)
    c_mutation_rate = ctypes.c_double(mutation_rate)
    c_elitism = ctypes.c_int(elitism)

    # 5. Buffers de sortie
    c_output_solution = (ctypes.c_int * MAX_SOLUTION_LENGTH)()
    c_output_length = ctypes.c_int()

    # --- Appel de la fonction C ---
    # Utiliser actual_population_size qui est maintenant égal à population_size (ou tronqué)
    final_fitness = ag_optimize_func(
        c_initial_population, c_solution_lengths,
        ctypes.c_int(actual_population_size), # Passer la taille effective
        ctypes.c_int(num_clients),
        c_orders, c_q, c_omega,
        c_distance_matrix, c_matrix_size,
        c_generations, c_mutation_rate, c_elitism,
        c_output_solution, ctypes.byref(c_output_length)
    )

    # --- Traitement du résultat ---
    best_solution_length = c_output_length.value
    best_solution = list(c_output_solution[:best_solution_length])

    return best_solution, float(final_fitness)
