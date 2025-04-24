#ifndef TABU_H
#define TABU_H

double tabu_optimize(
    int* initial_solution, int length,
    double* distance_matrix, int matrix_size,
    int* orders, int q, double omega,
    int iterations, int tabu_tenure, int neighborhood_size,
    int* best_solution_out, int* best_length_out
);

#endif
