import ctypes
import numpy as np
import os

# Charger la bibliothèque partagée
lib = ctypes.CDLL(os.path.abspath("rs_c.cpython-313-x86_64-linux-gnu.so"))

# Définir les types
lib.rs_optimize.argtypes = [
    ctypes.POINTER(ctypes.c_int), ctypes.c_int,                      # initial_solution, length
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,                   # positions (x1, y1, x2, y2, ...), n_positions
    ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_double,     # orders, q, omega
    ctypes.POINTER(ctypes.c_double), ctypes.c_int,                   # distance_matrix (row-major), size
    ctypes.c_int, ctypes.c_double, ctypes.c_double, ctypes.c_double, # iterations, T, Tmin, alpha
    ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)       # output_solution, output_length
]
lib.rs_optimize.restype = ctypes.c_double

def rs_c_optimize(state, initial_solution, distance_matrix, iterations, T=1000, T_min=0.1, alpha=0.99):
    MAX_LEN = 256

    # Initial solution
    init_sol = (ctypes.c_int * MAX_LEN)(*initial_solution + [0]*(MAX_LEN - len(initial_solution)))
    init_len = ctypes.c_int(len(initial_solution))

    # Positions
    flat_positions = []
    for x, y in state["position"]:
        flat_positions.extend([x, y])
    pos_arr = (ctypes.c_double * (2 * len(state["position"])))(*flat_positions)
    num_pos = ctypes.c_int(len(state["position"]))

    # Orders
    orders = (ctypes.c_int * num_pos.value)(*state["orders"])
    q = ctypes.c_int(state["q"])
    omega = ctypes.c_double(state["omega"])

    # Distance matrix
    flat_dm = distance_matrix.flatten()
    dm_size = ctypes.c_int(len(state["position"]))
    dm = (ctypes.c_double * (dm_size.value * dm_size.value))(*flat_dm)

    # Output buffers
    out_sol = (ctypes.c_int * MAX_LEN)()
    out_len = ctypes.c_int()

    # Call C function
    final_fitness = lib.rs_optimize(
        init_sol, init_len,
        pos_arr, num_pos,
        orders, q, omega,
        dm, dm_size,
        ctypes.c_int(iterations),
        ctypes.c_double(T), ctypes.c_double(T_min), ctypes.c_double(alpha),
        out_sol, ctypes.byref(out_len)
    )

    return list(out_sol[:out_len.value]), final_fitness
