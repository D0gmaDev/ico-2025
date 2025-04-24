#ifndef RS_H
#define RS_H

double rs_optimize(
    int* initial_solution, int initial_length,
    double* positions, int num_positions,
    int* orders, int q, double omega,
    double* distance_matrix, int matrix_size,
    int iterations, double T, double T_min, double alpha,
    int* output_solution, int* output_length
);

#endif
