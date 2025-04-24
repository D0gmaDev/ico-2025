#ifndef COMMON_H
#define COMMON_H

#define MAX_SOLUTION_LENGTH 256

double distance(int i, int j, double* matrix, int n);

double compute_fitness(int* solution, int length, int* orders, double* matrix, int n, int q, double omega);

#endif
