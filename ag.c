#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include "ag.h"
#include "ico.h"

// --- Fonctions Utilitaires ---

// Fonction pour échanger deux entiers
void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

// Structure pour aider au tri (associe fitness et index)
typedef struct {
    double fitness;
    int index;
} FitnessIndex;

// Fonction de comparaison pour qsort (tri ascendant par fitness)
int compareFitnessIndex(const void* a, const void* b) {
    FitnessIndex* fa = (FitnessIndex*)a;
    FitnessIndex* fb = (FitnessIndex*)b;
    if (fa->fitness < fb->fitness) return -1;
    if (fa->fitness > fb->fitness) return 1;
    return 0;
}

// --- Opérateurs Génétiques ---

// Fonction Crossover (basée sur le PMX simplifié et adapté de AG.py)
// Prend les pointeurs directs vers les solutions parents et l'enfant pour modification
void crossover_c(const int* parent1, int len1, const int* parent2, int len2,
    int* child, int* child_len, int num_clients) {
int size = (len1 < len2) ? len1 : len2;
if (size <= 2) {
memcpy(child, parent1, len1 * sizeof(int));
*child_len = len1;
return;
}
*child_len = 0; // Réinitialiser la longueur de l'enfant

int segment_start = 1 + rand() % (size - 2);
int segment_end = 1 + rand() % (size - 2);
if (segment_start == segment_end) segment_end = (segment_start + 1) % (size - 1);
if (segment_start > segment_end) swap(&segment_start, &segment_end);
segment_end++;

int present_clients[num_clients + 1];
for (int i = 0; i <= num_clients; ++i) present_clients[i] = 0;

child[(*child_len)++] = 0; // Ajouter le dépôt de départ

// Copier le segment du parent1
for (int i = segment_start; i < segment_end; ++i) {
if (parent1[i] > 0 && parent1[i] <= num_clients && !present_clients[parent1[i]]) {
child[(*child_len)++] = parent1[i];
present_clients[parent1[i]] = 1;
}
}

// Ajouter les clients du parent2 qui ne sont pas encore présents
for (int i = 1; i < len2 - 1; ++i) {
if (parent2[i] > 0 && parent2[i] <= num_clients && !present_clients[parent2[i]]) {
child[(*child_len)++] = parent2[i];
present_clients[parent2[i]] = 1;
}
}

// Ajouter les clients manquants (qui n'étaient ni dans le segment de P1 ni dans P2)
for (int i = 1; i <= num_clients; ++i) {
if (!present_clients[i]) {
child[(*child_len)++] = i;
}
}

// S'assurer que la longueur ne dépasse pas la limite et ajouter le dépôt de fin
if (*child_len < MAX_SOLUTION_LENGTH - 1) {
child[(*child_len)++] = 0;
} else {
child[MAX_SOLUTION_LENGTH - 1] = 0;
*child_len = MAX_SOLUTION_LENGTH;
fprintf(stderr, "Avertissement: Longueur de l'enfant après crossover a dépassé la limite.\n");
}
}

// Mutation par inversion (basée sur inversion_mutation de AG.py)
void inversion_mutation_c(int* solution, int length, double mutation_rate) {
    if (((double)rand() / RAND_MAX) < mutation_rate && length > 3) {
        int start = 1 + rand() % (length - 2); // Index entre 1 et length-2 inclus
        int end = 1 + rand() % (length - 2);
        if (start == end) {
             end = (start + 1) % (length - 1); // Assurer start != end
             if (end == 0) end = 1; // Éviter l'index 0 si length est petit
        }
        if (start > end) swap(&start, &end);

        // Inverser le segment [start, end] (inclusif)
        while (start < end) {
            swap(&solution[start], &solution[end]);
            start++;
            end--;
        }
    }
}

// Sélection par tournoi (basée sur tournament_selection de AG.py)
// Retourne l'INDEX du meilleur individu du tournoi dans la population actuelle
int tournament_selection_c(int population_size, const double* fitness_values, int tournament_size) {
    if (tournament_size > population_size) tournament_size = population_size;
    if (tournament_size <= 0) tournament_size = 1;

    int best_idx = -1;
    double best_fitness = INFINITY; // Commencer avec une fitness infinie

    for (int i = 0; i < tournament_size; ++i) {
        int idx = rand() % population_size;
        if (fitness_values[idx] < best_fitness) {
            best_fitness = fitness_values[idx];
            best_idx = idx;
        }
    }
    // Gérer le cas où tous les participants ont une fitness infinie (peu probable)
    if (best_idx == -1) {
        best_idx = rand() % population_size;
    }
    return best_idx;
}


// --- Fonction Principale de l'AG ---

double ag_c_optimize(
    int* initial_population, int* solution_lengths,
    int population_size, int num_clients,
    int* orders, int q, double omega,
    double* distance_matrix, int matrix_size,
    int generations, double mutation_rate, int elitism,
    int* output_solution, int* output_length)
{
    srand(time(NULL)); // Initialiser le générateur de nombres aléatoires

    // --- Allocation Mémoire pour la population et les données associées ---
    // Utilisation de tableaux statiques basés sur les defines de ag.h
    // Attention : si population_size > MAX_POPULATION_SIZE ou solution_lengths > MAX_SOLUTION_LENGTH,
    // cela causera un débordement de buffer.
    if (population_size > MAX_POPULATION_SIZE) {
        fprintf(stderr, "Erreur: population_size (%d) dépasse MAX_POPULATION_SIZE (%d)\n", population_size, MAX_POPULATION_SIZE);
        return -1.0; // Indiquer une erreur
    }

    int current_population[MAX_POPULATION_SIZE][MAX_SOLUTION_LENGTH];
    int current_lengths[MAX_POPULATION_SIZE];
    double current_fitness[MAX_POPULATION_SIZE];

    int new_population[MAX_POPULATION_SIZE][MAX_SOLUTION_LENGTH];
    int new_lengths[MAX_POPULATION_SIZE];
    double new_fitness[MAX_POPULATION_SIZE];

    int best_solution_overall[MAX_SOLUTION_LENGTH];
    int best_length_overall = 0;
    double best_fitness_overall = INFINITY;

    // --- Initialisation de la population ---
    int current_pop_offset = 0;
    for (int i = 0; i < population_size; ++i) {
        if (solution_lengths[i] > MAX_SOLUTION_LENGTH) {
             fprintf(stderr, "Erreur: solution_lengths[%d] (%d) dépasse MAX_SOLUTION_LENGTH (%d)\n", i, solution_lengths[i], MAX_SOLUTION_LENGTH);
             return -1.0; // Indiquer une erreur
        }
        memcpy(current_population[i], initial_population + current_pop_offset, solution_lengths[i] * sizeof(int));
        current_lengths[i] = solution_lengths[i];
        current_pop_offset += solution_lengths[i];

        // Calculer le fitness initial
        current_fitness[i] = compute_fitness(current_population[i], current_lengths[i], orders, distance_matrix, matrix_size, q, omega);

        // Mettre à jour la meilleure solution globale initiale
        if (current_fitness[i] < best_fitness_overall) {
            best_fitness_overall = current_fitness[i];
            memcpy(best_solution_overall, current_population[i], current_lengths[i] * sizeof(int));
            best_length_overall = current_lengths[i];
        }
    }

     // --- Boucle Principale des Générations ---
    for (int gen = 0; gen < generations; ++gen) {

        // --- Tri de la population actuelle par fitness ---
        FitnessIndex fitness_idx_map[population_size];
        for(int i=0; i<population_size; ++i) {
            fitness_idx_map[i].fitness = current_fitness[i];
            fitness_idx_map[i].index = i;
        }
        qsort(fitness_idx_map, population_size, sizeof(FitnessIndex), compareFitnessIndex);

        // --- Création de la nouvelle population ---
        int new_pop_count = 0;

        // 1. Élitism: Copier les meilleurs individus
        for (int i = 0; i < elitism && i < population_size; ++i) {
             int elite_idx = fitness_idx_map[i].index;
             memcpy(new_population[new_pop_count], current_population[elite_idx], current_lengths[elite_idx] * sizeof(int));
             new_lengths[new_pop_count] = current_lengths[elite_idx];
             new_fitness[new_pop_count] = current_fitness[elite_idx];
             new_pop_count++;
        }

        // 2. Remplir le reste de la population
        while (new_pop_count < population_size) {
            // Sélectionner les parents (utilisation de tournament_selection comme dans AG.py)
            int parent1_idx = tournament_selection_c(population_size, current_fitness, 5); // tournament_size=5
            int parent2_idx = tournament_selection_c(population_size, current_fitness, 5);
             // Éviter de croiser un individu avec lui-même (simple vérification)
            if (population_size > 1) {
                 while(parent2_idx == parent1_idx) {
                      parent2_idx = tournament_selection_c(population_size, current_fitness, 5);
                 }
            }

            // Créer l'enfant
            int child_solution[MAX_SOLUTION_LENGTH];
            int child_length;

            crossover_c(current_population[parent1_idx], current_lengths[parent1_idx],
                        current_population[parent2_idx], current_lengths[parent2_idx],
                        child_solution, &child_length, num_clients);

            // Appliquer la mutation (utilisation de inversion_mutation comme dans AG.py)
            inversion_mutation_c(child_solution, child_length, mutation_rate);

             // S'assurer que la solution mutée est valide (commence et finit par 0)
             if (child_length > 0) {
                child_solution[0] = 0;
                child_solution[child_length - 1] = 0;
             } else { // Gérer le cas d'une solution vide/invalide après mutation/crossover
                 child_length = 2; // Solution minimale [0, 0]
                 child_solution[0] = 0;
                 child_solution[1] = 0;
             }


            // Calculer le fitness de l'enfant
            double child_fitness = compute_fitness(child_solution, child_length, orders, distance_matrix, matrix_size, q, omega);

             // Ajouter l'enfant à la nouvelle population
             // pour des raisons de performance et de complexité en C.
             if (child_length <= MAX_SOLUTION_LENGTH) {
                 memcpy(new_population[new_pop_count], child_solution, child_length * sizeof(int));
                 new_lengths[new_pop_count] = child_length;
                 new_fitness[new_pop_count] = child_fitness;
                 new_pop_count++;

                 // Mettre à jour la meilleure solution globale si l'enfant est meilleur
                 if (child_fitness < best_fitness_overall) {
                     best_fitness_overall = child_fitness;
                     memcpy(best_solution_overall, child_solution, child_length * sizeof(int));
                     best_length_overall = child_length;
                 }
             } else {
                   fprintf(stderr, "Avertissement: Longueur de l'enfant (%d) dépasse MAX_SOLUTION_LENGTH (%d) et n'a pas été ajouté.\n", child_length, MAX_SOLUTION_LENGTH);
             }
        } // Fin du remplissage de la nouvelle population

        // --- Remplacer l'ancienne population par la nouvelle ---
        for (int i = 0; i < population_size; ++i) {
             memcpy(current_population[i], new_population[i], new_lengths[i] * sizeof(int));
             current_lengths[i] = new_lengths[i];
             current_fitness[i] = new_fitness[i];
        }

    }

    // --- Copier le résultat final dans les buffers de sortie ---
    if (best_length_overall > 0 && best_length_overall <= MAX_SOLUTION_LENGTH) {
        memcpy(output_solution, best_solution_overall, best_length_overall * sizeof(int));
        *output_length = best_length_overall;
    } else {
         // Si aucune solution valide n'a été trouvée (très improbable) ou si la longueur dépasse la limite
         // Retourner une solution minimale [0, 0]
         output_solution[0] = 0;
         output_solution[1] = 0;
         *output_length = 2;
    }


    return best_fitness_overall;
}
