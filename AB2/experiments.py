import matplotlib.pyplot as plt
import linear_solvers as linsol
import poisson_problem as pp
from block_matrix import BlockMatrix
import numpy as np

def test_1(max_n, slices_n, max_d, slices_d):
    n_list = np.logspace(2, max_n, slices_n, dtype=int)
    d_list = np.logspace(1, max_d, slices_d, dtype=int)
    
    hat_u = pp.bsp_1
    u = pp.pp_zu_bsp_1
    
    for n in n_list:
        for d in d_list:
            errors += pp.compute_error(d, n, hat_u, u)
