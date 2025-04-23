#ifndef RS_H
#define RS_H

#define MAX_SOLUTION_LENGTH 256
#define MAX_POSITIONS 256

double rs_optimize(
    int* initial_solution, int initial_length,
    double* positions, int num_positions,
    int* orders, int q, double omega,
    double* distance_matrix, int matrix_size,
    int iterations, double T, double T_min, double alpha,
    int* output_solution, int* output_length
);

#endif
