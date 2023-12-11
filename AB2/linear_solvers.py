import numpy as np
import scipy
import block_matrix as bm

def solve_lu(p : np.ndarray, l : np.ndarray, u : np.ndarray, b : np.ndarray):
    """ Solves the linear system Ax = b via forward and backward substitution
    given the decomposition A = p * l * u.

    Parameters
    ----------
    p : numpy.ndarray
        permutation matrix of LU-decomposition
    l : numpy.ndarray
        lower triangular unit diagonal matrix of LU-decomposition
    u : numpy.ndarray
        upper triangular matrix of LU-decomposition
    b : numpy.ndarray
        vector of the right-hand-side of the linear system

    Returns
    -------
    x : numpy.ndarray
        solution of the linear system
    """
    p_t = p.transpose() #p.t geht vlt auch
    z = p_t * b
    #Lösen von Ly = z rekrusiv
    y = [0] * len(b)    #np.ndarray(shape=(0,len(b)))
    y[0] = z[0]/ l[0][0]
    for i in range(1,len(b)):       #vlt len(b)-1 ??    ->bestimmt y_2, .... ,y()
        old = 0
        for n in range(i):      #berechnet eine Summe der rechnung
            old = (l[i][n] * y[n]) + old
        y[i] = (z[i] + old) / l[i][i]

    x = [0] * len(b)    #np.ndarray(shape=(0,len(b)))
    m = len(b)-1
    x[m] = y[m]/ u[m][m]
    for i in range(m-1,-1,-1):
        old=0
        for n in range(m,i,-1):
            old = u[i][n]*x[n]
        x[i] = (y[i]+old)/u[i][i]
    return x
    #jetzt sollten wir den Vektor y berechnet haben
    #als nächstes lösen wir u * x = y rekrusiv

x = np.ndarray([1,2,3])

print(x)

mat_1 = bm.BlockMatrix(2, 3)
solve = solve_lu(mat_1.get_lu()[0], mat_1.get_lu()[1], mat_1.get_lu()[2], [1, 1, 1, 1])
print(solve)