#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "ico.h"
#include "rs.h"

void mutate(int* solution, int length, int* new_solution, int* new_length) {
    memcpy(new_solution, solution, length * sizeof(int));
    *new_length = length;

    double r = (double) rand() / RAND_MAX;
    if (r < 0.8) {
        int i = rand() % length;
        int j = rand() % length;
        while (j == i) j = rand() % length;

        int temp = new_solution[i];
        new_solution[i] = new_solution[j];
        new_solution[j] = temp;
    } else if (r < 0.9 && length < MAX_SOLUTION_LENGTH) {
        int i = 1 + rand() % (length - 1);
        for (int j = length; j > i; j--) {
            new_solution[j] = new_solution[j - 1];
        }
        new_solution[i] = 0;
        *new_length += 1;
    } else {
        for (int i = 0; i < length; i++) {
            if (new_solution[i] == 0 && i != 0 && i != length - 1) {
                for (int j = i; j < length - 1; j++) {
                    new_solution[j] = new_solution[j + 1];
                }
                *new_length -= 1;
                break;
            }
        }
    }
}

double rs_optimize(
    int* initial_solution, int initial_length,
    double* positions, int num_positions,
    int* orders, int q, double omega,
    double* distance_matrix, int matrix_size,
    int iterations, double T, double T_min, double alpha,
    int* output_solution, int* output_length
) {
    srand(time(NULL));

    int current_solution[MAX_SOLUTION_LENGTH];
    memcpy(current_solution, initial_solution, initial_length * sizeof(int));
    int current_length = initial_length;
    double current_fitness = compute_fitness(current_solution, current_length, orders, distance_matrix, matrix_size, q, omega);

    int best_solution[MAX_SOLUTION_LENGTH];
    memcpy(best_solution, current_solution, current_length * sizeof(int));
    int best_length = current_length;
    double best_fitness = current_fitness;

    int new_solution[MAX_SOLUTION_LENGTH];
    int new_length;

    while (T > T_min) {
        for (int i = 0; i < iterations; i++) {
            mutate(current_solution, current_length, new_solution, &new_length);
            double new_fitness = compute_fitness(new_solution, new_length, orders, distance_matrix, matrix_size, q, omega);

            if (new_fitness < current_fitness) {
                memcpy(current_solution, new_solution, new_length * sizeof(int));
                current_length = new_length;
                current_fitness = new_fitness;

                if (new_fitness < best_fitness) {
                    memcpy(best_solution, new_solution, new_length * sizeof(int));
                    best_length = new_length;
                    best_fitness = new_fitness;
                }
            } else {
                double delta = new_fitness - current_fitness;
                double prob = exp(-delta / T);
                if (((double) rand() / RAND_MAX) < prob) {
                    memcpy(current_solution, new_solution, new_length * sizeof(int));
                    current_length = new_length;
                    current_fitness = new_fitness;
                }
            }
        }
        T *= alpha;
    }

    memcpy(output_solution, best_solution, best_length * sizeof(int));
    *output_length = best_length;
    return best_fitness;
}
