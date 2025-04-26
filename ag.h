#ifndef AG_H_
#define AG_H_

#include <stdlib.h>

#define MAX_POPULATION_SIZE 150

double ag_c_optimize(
    int* initial_population,
    int* solution_lengths,
    int population_size,
    int num_clients,
    int* orders, int q, double omega,
    double* distance_matrix, int matrix_size,
    int generations, double mutation_rate, int elitism,
    int* output_solution,
    int* output_length
);

#endif
