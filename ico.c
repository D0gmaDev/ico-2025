#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ico.h"

double distance(int i, int j, double* matrix, int n) {
    return matrix[i * n + j];
}

double compute_fitness(int* solution, int length, int* orders, double* matrix, int n, int q, double omega) {
    double total = 0.0;
    int load = 0;
    int prev = solution[0];
    int penalty_count = 0;
    int trucks = 0;
    int delivered[n];
    for (int i = 0; i < n; i++) {
        delivered[i] = 0;  // Initialize delivered array
    }

    for (int i = 1; i < length; i++) {
        int curr = solution[i];

        // Check if we're back at the depot (0)
        if (curr == 0) {
            trucks++;
            load = 0;
        } else {
            // Capacity handling
            if (load + orders[curr] > q) {
                penalty_count += (orders[curr] - (q - load) + 1) / 2;  // Not enough capacity
                load = orders[curr];  // Start a new trip
            } else {
                load += orders[curr];
            }
            delivered[curr] = 1;  // Mark as delivered
        }

        // Total distance calculation
        total += distance(prev, curr, matrix, n);
        prev = curr;

        // Check if the load exceeds capacity
        if (load > q) {
            return 1e9;  // Exceeded capacity, invalid solution
        }
    }

    // Penalty for missing deliveries
    for (int i = 1; i < n; i++) {
        if (!delivered[i]) {
            penalty_count += 5;  // Penalize undelivered orders
        }
    }

    // Ensure the path starts and ends at depot
    if (solution[0] != 0 || solution[length - 1] != 0) {
        penalty_count += 2;  // Route does not start/end at depot
    }

    // Penalty for excess trucks
    double trucks_value = trucks * omega;
    double penalty_value = penalty_count * 1000.0;  // Penalty multiplier

    return total + trucks_value + penalty_value;
}