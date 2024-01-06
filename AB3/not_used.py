import numpy as np
from block_matrix import BlockMatrix
import poisson_problem as pp
from scipy import sparse, linalg
import linear_solvers as linsol
from plotter import plotter

def add_row_to_row(mat, a, b, value = 1):
    new_mat = mat
    ident = sparse.eye(mat.shape[0]).tolil()
    ident[a,b]=value
    new_mat = ident.dot(new_mat)
    return new_mat

def sparse_swapper(mat, a, b, mode="row"):
    """
    Reorders the rows and/or columns in a scipy sparse matrix to the specified order.
    """
    if mode!="row" and mode!="col":
        raise ValueError("mode must be 'row' or 'col'!")
    if max(a,b) > mat.shape[0]:
        raise ValueError("a and b must relate to rows/columns in the matrix!")

    new_order = [x for x in range(a)] + [b] + [x for x in range(a+1, b)] + [a]
    new_order += [x for x in range(b+1, mat.shape[0])]

    new_mat = mat
    if mode == "row":
        ident = sparse.eye(mat.shape[0]).tocoo()
        ident.row = ident.row[new_order]
        new_mat = ident.dot(new_mat)
    if mode == "col":
        ident = sparse.eye(mat.shape[1]).tocoo()
        ident.col = ident.col[new_order]
        new_mat = new_mat.dot(ident)
    return new_mat

def get_lu(self):
    """ Provides an LU-Decomposition of the represented matrix A of the form A = p * l * u
    Returns
    -------
    p : numpy.ndarray
        permutation matrix of LU-decomposition
    l : numpy.ndarray
        lower triangular unit diagonal matrix of LU-decomposition
    u : numpy.ndarray
        upper triangular matrix of LU-decomposition
    """
    return linalg.lu(self.a_d.toarray(), permute_l=False)
    # Since these matrices have no entries equal to 0 on the main diagonal no swaps are needed
    #U = self.a_d.tocsc()
    #entry_list = []
    #for k in range(self.a_d.shape[0]):
    #    for l in range(k + 1, self.a_d.shape[0]):
    #        mul = U[l, k] / U[k, k]
    #        entry_list += [(k, l, mul)]
    #        U = add_row_to_row(U, l, k, -mul)

FAST_MODE = True

def graph_error_helper(d: int , n : int, pp_u : callable, u: callable):
    h = 1/n
    # create coefficient matrix A for given n and d
    mat = BlockMatrix(d,n)
    mat_p, mat_l, mat_u = mat.get_lu()  #pylint: disable=invalid-name, disable=unbalanced-tuple-unpacking

    #b_vector = pp.rhs(n,d,pp.pp_zu_bsp_1)
    b_vector = []
    for i in range(1, (n-1)**d+1):
        # create list of discretization points
        x = pp.inv_idx(i,d,n)  #pylint: disable=invalid-name
        x = [j/n for j in x]    #pylint: disable=invalid-name
        #calculate right side of f(x)*(-h^2)=b
        b_vector.append(pp_u(x)*(-h**2))

    # use fast mode or not as specified above
    if FAST_MODE:
        loesung = linsol.solve_lu(mat_p, mat_l, mat_u, b_vector)
    else:
        loesung = linsol.solve_lu_alt(mat_p, mat_l, mat_u, b_vector)

    maximum = pp.compute_error(d,n,loesung, u)
    return maximum


def graph_error(n_max : int, pp_u : callable, u: callable):
    x_values = [[],[],[]]
    y_values = [[],[],[]]
    n = np.logspace(0.4, n_max, 20, dtype=int)
    n = [int(e) for e in n]
    labels = []
    for d in [1,2,3]:
        for e in n:
            print(e)
            x = (e-1)**d
            y = graph_error_helper(d,e, pp_u, u)
            x_values[d-1].append(x)
            y_values[d-1].append(y)
        labels += [f"Maximalfehler d={d}"]

    plotter(x_values, y_values, labels, ["dashdot"]*3,["b", "r", "c"])
