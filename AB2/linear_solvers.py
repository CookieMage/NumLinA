import numpy as np
import scipy

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
    y = np.ndarray(shape=(0,len(b)))
    y[0] = z[0]/ l[0][0]
    for i in range(1,len(b)):       #vlt len(b)-1 ??    ->bestimmt y_2, .... ,y()
        old = 0
        for n in range(0,i-1):      #berechnet eine Summe der rechnung
            old = ([i,n] * y[n]) + old
        y[i] = (z[i] + old) / l[i][i]

    x = np.ndarray(shape=(0,len(b)))
    m = len(b)-1
    x[m] = y[m]/ u[m][m]
    for i in range(m-1,0,-1):
        old=0
        for n in range(m,i+1,-1):
            old = u[i][n]*x[n]
        x[i] = (y[i]+old)/u[i][i]
    #jetzt sollten wir den Vektor y berechnet haben
    #als nächstes lösen wir u * x = y rekrusiv








def solve_sor(A, b, x0,
              params=dict(eps=1e-8, max_iter=1000, var_x=1e-4),
              omega=1.5):
    """ Solves the linear system Ax = b via the successive over relaxation method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray (of shape (N,) )
        right-hand-side of the linear system
    x0 : numpy.ndarray (of shape (N,) )
        initial guess of the solution

    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        var_x : float
            minimal change of the iterate in every step in the infinity norm. If set
            less or equal to 0 no constraint on the change is imposed.
        omega : float, optional
            relaxation parameter

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list (of numpy.ndarray of shape (N,) )
        iterates of the algorithm. First entry is `x0`.
    list (of float)
        infinity norm of the residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., `eps=0` and `max_iter=0`, etc.
    """


def solve_gs(A, b, x0,
             params=dict(eps=1e-8, max_iter=1000, var_x=1e-4)):
    """ Solves the linear system Ax = b via the Jacobi method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray (of shape (N,) )
        right-hand-side of the linear system
    x0 : numpy.ndarray (of shape (N,) )
        initial guess of the solution

    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        var_x : float
            minimal change of the iterate in every step in the infinity norm. If set
            less or equal to 0 no constraint on the change is imposed.

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list (of numpy.ndarray of shape (N,) )
        iterates of the algorithm. First entry is `x0`.
    list (of float)
        infinity norm of the residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., `eps=0` and `max_iter=0`, etc.
    """


def solve_es(A, b, x0, params=dict(eps=1e-8, max_iter=1000, var_x=1e-4)):
    """ Solves the linear system Ax = b via the Gauss-Seidel method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray (of shape (N,) )
        right-hand-side of the linear system
    x0 : numpy.ndarray (of shape (N,) )
        initial guess of the solution

    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        var_x : float
            minimal change of the iterate in every step in the infinity norm. If set
            less or equal to 0 no constraint on the change is imposed.

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list (of numpy.ndarray of shape (N,) )
        iterates of the algorithm. First entry is `x0`.
    list (of float)
        infinitiy norm of the residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., `eps=0` and `max_iter=0`, etc.
    """


def solve_cg(A, b, x0,
             params=dict(eps=1e-8, max_iter=1000, var_x=1e-4)):
    """ Solves the linear system Ax = b via the conjugated gradient method.

    Parameters
    ----------
    A : scipy.sparse.csr_matrix
        system matrix of the linear system
    b : numpy.ndarray (of shape (N,) )
        right-hand-side of the linear system
    x0 : numpy.ndarray (of shape (N,) )
        initial guess of the solution

    params : dict, optional
        dictionary containing termination conditions

        eps : float
            tolerance for the norm of the residual in the infinity norm. If set
            less or equal to 0 no constraint on the norm of the residual is imposed.
        max_iter : int
            maximal number of iterations that the solver will perform. If set
            less or equal to 0 no constraint on the number of iterations is imposed.
        var_x : float
            minimal change of the iterate in every step in the infinity norm. If set
            less or equal to 0 no constraint on the change is imposed.

    Returns
    -------
    str
        reason of termination. Key of the respective termination parameter.
    list (of numpy.ndarray of shape (N,) )
        iterates of the algorithm. First entry is `x0`.
    list (of float)
        residuals of the iterates

    Raises
    ------
    ValueError
        If no termination condition is active, i.e., `eps=0` and `max_iter=0`, etc.
    """