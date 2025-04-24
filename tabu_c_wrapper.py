import ctypes
import numpy as np
from numpy.ctypeslib import ndpointer

lib = ctypes.CDLL("./tabu.cpython-313-x86_64-linux-gnu.so")  # Compile with setup.py or manually with gcc

lib.tabu_optimize.argtypes = [
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # initial_solution
    ctypes.c_int,
    ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),  # distance_matrix
    ctypes.c_int,
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # orders
    ctypes.c_int,  # q
    ctypes.c_double,  # omega
    ctypes.c_int,  # iterations
    ctypes.c_int,  # tabu_tenure
    ctypes.c_int,  # neighborhood_size
    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),  # output_solution
    ctypes.POINTER(ctypes.c_int),  # output_length
]

lib.tabu_optimize.restype = ctypes.c_double

def tabu_c_optimize(state, initial_solution, distance_matrix, iterations=500, tabu_tenure=7, neighborhood_size=25):
    orders = state["orders"]
    n = len(orders)
    initial_solution = np.array(initial_solution, dtype=np.int32)
    orders = np.array(orders, dtype=np.int32)
    distance_matrix = np.array(distance_matrix, dtype=np.float64)

    output_solution = np.zeros_like(initial_solution)
    output_length = ctypes.c_int()

    best_cost = lib.tabu_optimize(
        initial_solution,
        len(initial_solution),
        distance_matrix,
        n,
        orders,
        state["q"],
        state["omega"],
        iterations,
        tabu_tenure,
        neighborhood_size,
        output_solution,
        ctypes.byref(output_length)
    )

    return output_solution[:output_length.value].tolist(), best_cost
