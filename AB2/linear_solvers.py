'''Gruppe: 21

program for solving linear systems of shape Ax=b for x

functions
---------
solve_lu_alt()
    solves the linear system Ax = b via forward and backward substitution
    given the decomposition A = p * l * u
solve_lu()
    solves the linear system Ax = b via forward and backward substitution
    given the decomposition A = p * l * u
main()
    Example of code that can be run using the provided class and methods
'''

import numpy as np
from scipy import linalg

def solve_lu_alt(p : np.ndarray, l : np.ndarray, u : np.ndarray, b : np.ndarray):   #pylint: disable=invalid-name
    """ alternative function that solves the linear system Ax = b via forward and backward substitution
    given the decomposition A = p * l * u

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
    p_t = p.transpose()
    z = p_t.dot(b)  #pylint: disable=invalid-name

    #Loesen von Ly = z rekrusiv
    y = [0] * len(b)    #pylint: disable=invalid-name
    y[0] = z[0]/ l[0][0]
    for i in range(1,len(b)):
        old = 0
        for n in range(i):  #pylint: disable=invalid-name
            old = (-l[i][n] * y[n]) + old
        y[i] = (z[i] + old) / l[i][i]

    x = [0] * len(b)    #pylint: disable=invalid-name
    m = len(b)-1    #pylint: disable=invalid-name
    x[m] = y[m]/ u[m][m]
    for i in range(m-1,-1,-1):
        old=0
        for n in range(m,i,-1): #pylint: disable=invalid-name
            old = (-u[i][n]*x[n]) + old
        x[i] = (y[i]+old)/u[i][i]
    return np.array(x)

def solve_lu(p : np.ndarray, l : np.ndarray, u : np.ndarray, b : np.ndarray):   #pylint: disable=invalid-name
    """ solves the linear system Ax = b via forward and backward substitution
    given the decomposition A = p * l * u

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
    p_t = p.transpose()
    z = p_t.dot(b)  #pylint: disable=invalid-name

    y = linalg.solve_triangular(l,z,trans = 0,lower = True) #pylint: disable=invalid-name
    x = linalg.solve_triangular(u,y,trans=0,lower = False)  #pylint: disable=invalid-name
    return np.array(x)


def main():
    '''Example of code that can be run using the provided class and methods
    '''
    # p = np.array([[0,0,0,1],
    #               [0,0,1,0],
    #               [1,0,0,0],
    #               [0,1,0,0]])
    # b = np.array([-10,1,-8,-8,])
    # u= np.array([[12,4,4,4],
    #              [0,12,0,-8],
    #              [0,0,-4,8],
    #              [0,0,0,-8]])
    # l= np.array([[1,0,0,0],
    #              [0,1,0,0],
    #              [1/4,1/2,1,0],
    #              [1/2,1/4,-1/4,1]])
    true = True
    if true:
        p = np.array([[1,0,0],  #pylint: disable=invalid-name
                     [0,0,1],
                     [0,1,0]])
        b= np.array([1,1,1])    #pylint: disable=invalid-name
        u = np.array([[3,2,1],  #pylint: disable=invalid-name
                      [0,2,4],
                      [0,0,-6]])
        l = np.array([[1,0,0],  #pylint: disable=invalid-name
                      [7,1,0],
                      [9,2,1]])

    print("Loesung von uns" , solve_lu_alt(p, l, u, b))
    print("Loesung von scipy", solve_lu(p, l, u, b))
    f=[1.66666666666666, -1.6666666666 , -0.66666666666]    #pylint: disable=invalid-name
    test_a = np.dot(p,l)
    test_b = np.dot(test_a,u)
    test_c = np.dot(test_b,f)
    print(test_c)

if __name__ == "__main__":
    main()
