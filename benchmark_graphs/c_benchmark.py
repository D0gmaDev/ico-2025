import time
import matplotlib.pyplot as plt
import numpy as np

from ico import compute_distance_matrix, fitness
from RS import RS
from rs_c_wrapper import rs_c_optimize
from Tabou import tabu_search
from tabu_c_wrapper import tabu_c_optimize

# Sample state with positions and orders
state = {
    "position": [
        (0, 0), (5, 2), (3, -3), (2, 2), (1, 0),
        (3, 3), (2, 1), (-5, 0), (7, 1),
        (-3, -2), (5, 1), (1, 2), (4, 2)
    ],
    "orders": [0, 5, 10, 7, 8, 6, 9, 4, 3, 1, 2, 3, 7],
    "q": 400,
    "omega": 10
}

distance_matrix = compute_distance_matrix(state)
initial_solution = [0, 9, 7, 0, 2, 0, 4, 6, 10, 8, 1, 12, 5, 3, 11, 0]

# Parameters for Recuit Simulé
T = 1000
T_min = 0.1
alpha = 0.99

# Range of iterations to test
iteration_values_rs = [100, 200, 500, 1000, 2000, 5000]
iteration_values_tabou = [500, 1000, 5000, 10000, 20000, 40000]

rs_python_times = []
rs_c_times = []

tabu_python_times = []
tabu_c_times = []

# RS Benchmark
for iterations in iteration_values_rs:
    print(f"Running RS with {iterations} iterations...")

    start = time.time()
    RS(state, initial_solution, distance_matrix, iterations, T, T_min, alpha)
    rs_python_times.append(time.time() - start)

    start = time.time()
    rs_c_optimize(state, initial_solution, distance_matrix, iterations, T, T_min, alpha)
    rs_c_times.append(time.time() - start)

# Tabu Benchmark
for iterations in iteration_values_tabou:
    print(f"Running Tabu with {iterations} iterations...")

    start = time.time()
    tabu_search(state, initial_solution, distance_matrix, iterations)
    tabu_python_times.append(time.time() - start)

    start = time.time()
    tabu_c_optimize(state, initial_solution, distance_matrix, iterations)
    tabu_c_times.append(time.time() - start)

# Plot RS times with linear fits
plt.figure(figsize=(10, 6))
x_rs = np.array(iteration_values_rs)
y_rs_python = np.array(rs_python_times)
y_rs_c = np.array(rs_c_times)

# Fit linear models
slope_py_rs, intercept_py_rs = np.polyfit(x_rs, y_rs_python, 1)
slope_c_rs, intercept_c_rs = np.polyfit(x_rs, y_rs_c, 1)

# Plot data points
plt.plot(x_rs, y_rs_python, label=f'Python RS (slope={slope_py_rs:.2e})', marker='o', color='tab:blue')
plt.plot(x_rs, y_rs_c, label=f'C RS (slope={slope_c_rs:.2e})', marker='o', color='tab:orange')

# Plot fits
plt.plot(x_rs, slope_py_rs * x_rs + intercept_py_rs, linestyle='--', color='tab:blue')
plt.plot(x_rs, slope_c_rs * x_rs + intercept_c_rs, linestyle='--', color='tab:orange')

plt.xlabel('Iterations')
plt.ylabel('Execution Time (s)')
plt.title('RS: Python vs C Execution Time with Linear Fit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('benchmark_graphs/rs_execution_time.png')
plt.show()

# Plot Tabou times with linear fits
plt.figure(figsize=(10, 6))
x_tabu = np.array(iteration_values_tabou)
y_tabu_python = np.array(tabu_python_times)
y_tabu_c = np.array(tabu_c_times)

# Fit linear models
slope_py_tabu, intercept_py_tabu = np.polyfit(x_tabu, y_tabu_python, 1)
slope_c_tabu, intercept_c_tabu = np.polyfit(x_tabu, y_tabu_c, 1)

# Plot data points
plt.plot(x_tabu, y_tabu_python, label=f'Python Tabou (slope={slope_py_tabu:.2e})', marker='o', color='tab:blue')
plt.plot(x_tabu, y_tabu_c, label=f'C Tabou (slope={slope_c_tabu:.2e})', marker='o', color='tab:orange')

# Plot fits
plt.plot(x_tabu, slope_py_tabu * x_tabu + intercept_py_tabu, linestyle='--', color='tab:blue')
plt.plot(x_tabu, slope_c_tabu * x_tabu + intercept_c_tabu, linestyle='--', color='tab:orange')

plt.xlabel('Iterations')
plt.ylabel('Execution Time (s)')
plt.title('Tabou: Python vs C Execution Time with Linear Fit')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('benchmark_graphs/tabou_execution_time.png')
plt.show()