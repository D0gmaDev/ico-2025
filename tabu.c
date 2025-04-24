#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "ico.h"

typedef struct {
    int i;
    int j;
} Move;

int move_equals(Move a, Move b) {
    return (a.i == b.i && a.j == b.j) || (a.i == b.j && a.j == b.i);
}

int is_valid_solution(int* solution, int length, int* orders, int q) {
    int total_load = 0;
    for (int i = 0; i < length; i++) {
        if (solution[i] == 0) {
            total_load = 0;
        } else {
            total_load += orders[solution[i]];
            if (total_load > q) return 0;
        }
    }
    return 1;
}

double tabu_optimize(
    int* initial_solution, int length,
    double* distance_matrix, int matrix_size,
    int* orders, int q, double omega,
    int iterations, int tabu_tenure, int neighborhood_size,
    int* best_solution_out, int* best_length_out
) {
    srand(time(NULL));
    int best_solution[MAX_SOLUTION_LENGTH];
    int current_solution[MAX_SOLUTION_LENGTH];
    memcpy(best_solution, initial_solution, length * sizeof(int));
    memcpy(current_solution, initial_solution, length * sizeof(int));

    double best_cost = compute_fitness(best_solution, length, orders, distance_matrix, matrix_size, q, omega);
    Move tabu_list[tabu_tenure];
    int tabu_size = 0;

    for (int it = 0; it < iterations; it++) {
        double best_neighbor_cost = 1e9;
        int best_neighbor[MAX_SOLUTION_LENGTH];
        Move best_move;

        for (int n = 0; n < neighborhood_size; n++) {
            int i = rand() % (length - 2) + 1;
            int j = rand() % (length - 2) + 1;
            while (i == j) j = rand() % (length - 2) + 1;

            int neighbor[MAX_SOLUTION_LENGTH];
            memcpy(neighbor, current_solution, length * sizeof(int));
            int temp = neighbor[i];
            neighbor[i] = neighbor[j];
            neighbor[j] = temp;

            if (!is_valid_solution(neighbor, length, orders, q)) continue;

            double cost = compute_fitness(neighbor, length, orders, distance_matrix, matrix_size, q, omega);

            Move move = {i, j};
            int in_tabu = 0;
            for (int t = 0; t < tabu_size; t++) {
                if (move_equals(move, tabu_list[t])) {
                    in_tabu = 1;
                    break;
                }
            }

            if (!in_tabu || cost < best_cost) {
                if (cost < best_neighbor_cost) {
                    memcpy(best_neighbor, neighbor, length * sizeof(int));
                    best_neighbor_cost = cost;
                    best_move = move;
                }
            }
        }

        if (best_neighbor_cost == 1e9) continue;

        memcpy(current_solution, best_neighbor, length * sizeof(int));

        if (best_neighbor_cost < best_cost) {
            memcpy(best_solution, best_neighbor, length * sizeof(int));
            best_cost = best_neighbor_cost;
            tabu_size = 0;
        } else {
            if (tabu_size < tabu_tenure) {
                tabu_list[tabu_size++] = best_move;
            } else {
                for (int t = 1; t < tabu_tenure; t++) {
                    tabu_list[t - 1] = tabu_list[t];
                }
                tabu_list[tabu_tenure - 1] = best_move;
            }
        }
    }

    memcpy(best_solution_out, best_solution, length * sizeof(int));
    *best_length_out = length;
    return best_cost;
}
