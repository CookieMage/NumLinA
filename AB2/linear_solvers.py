import numpy as np
import scipy
from block_matrix import BlockMatrix

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
    z = p_t.dot(b)
    print(z)
    #Lösen von Ly = z rekrusiv
    y = [0] * len(b)    #np.ndarray(shape=(0,len(b)))
    y[0] = z[0]/ l[0][0]
    for i in range(1,len(b)):       #vlt len(b)-1 ??    ->bestimmt y_2, .... ,y()
        old = 0
        for n in range(i):      #berechnet eine Summe der rechnung
            old = (l[i][n] * y[n]) + old
        y[i] = (z[i] + old) / l[i][i]
    print(y)    
    y = np.array([8,-8,16,-8])
    x = [0] * len(b)    #np.ndarray(shape=(0,len(b)))
    m = len(b)-1
    x[m] = y[m]/ u[m][m]
    for i in range(m-1,-1,-1):
        old=0
        for n in range(m,i,-1):
            old = u[i][n]*x[n]
        x[i] = (y[i]+old)/u[i][i]
    return np.array(x)
    #jetzt sollten wir den Vektor y berechnet haben
    #als nächstes lösen wir u * x = y rekrusiv



p = np.array([[0,0,1,0],
              [0,0,0,1],
              [1,0,0,0],
              [0,1,0,0]])
b = np.array([-10,14,8,-8])
u= np.array([[12,4,4,4],
             [0,12,0,-8],
             [0,0,-4,8],
             [0,0,0,-8]])
l= np.array([[1,0,0,0],
             [0,1,0,0],
             [1/4,1/2,1,0],
             [1/2,1/4,-1/4,1]])

print(solve_lu(p, l, u, b))