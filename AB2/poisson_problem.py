import numpy as np

def rhs(d : int, n : int, f : callable):
    """ Computes the right-hand side vector `b` for a given function `f`.

    Parameters
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    f : callable
        Function right-hand-side of Poisson problem. The calling signature is
        `f(x)`. Here `x` is an array_like of `numpy`. The return value
        is a scalar.

    Returns
    -------
    numpy.ndarray
        Vector to the right-hand-side f.

    Raises
    ------
    ValueError
        If d < 1 or n < 2.
    """
    if not isinstance(d, int) or not isinstance(n, int) or not isinstance(f, callable):
        raise TypeError
    if d < 1 or n < 2:
        raise ValueError
    end = (n-1)**d
    
    nx = np.ndarray([0]*end)
    for i,e in enumerate(nx):
        e = inv_idx(i+1, d, n)
        nx[i] = f(e)
    return nx


def idx(nx : list, n : int):
    """ Calculates the number of an equation in the Poisson problem for
    a given discretization point.

    Parameters
    ----------
    nx : list of int
        Coordinates of a discretization point, multiplied by n.
    n : int
        Number of intervals in each dimension.
    
    Return
    ------
    int
        Number of the corresponding equation in the Poisson problem.
    """
    if not isinstance(nx, list) or not isinstance(n, int):
        raise TypeError
    for e in nx:
        if e > (n-1):
            raise ValueError
    num = nx[0]
    # Als Alternative wäre es hier nur möglich zu schreiben:
    # for i,e in enumerate(nx[1:],1):
    # Durch die Kopierung der Liste (nx[1:]) ist dies unnötige aufwendig.
    for i in range(1,len(nx)):
        num = num + (n-1)**i * (nx[i]-1)
    return num

def inv_idx(m : int, d : int, n : int):
    """ Calculates the coordinates of a discretization point for a
    given equation number of the Poisson problem.
    
    Parameters
    ----------
    m : int
        Number of an equation in the Poisson Problem
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    
    Return
    ------
    list of int
        Coordinates of the corresponding discretization point, multiplied by n.
    """
    if not isinstance(m, int) or not isinstance(d, int) or not isinstance(n, int):
        raise TypeError
    if m > (n-1)**d:
        raise ValueError
    m = m-1
    nx = [1] * d
    for i in range(len(nx),0,-1):
        nx[i-1] = 1 + (m // ((n-1)**(i-1)))
        m = m % (n-1)**(i-1)
    return nx

def compute_error(d : int, n : int, hat_u : np.ndarray, u : callable):
    """ Computes the error of the numerical solution of the Poisson problem
    with respect to the infinity-norm.

    Parameters
    ----------
    d : int
        Dimension of the space
    n : int
        Number of intersections in each dimension
    hat_u : array_like of 'numpy'
        Finite difference approximation of the solution of the Poisson problem
        at the discretization points
    u : callable
        Solution of the Poisson problem
        The calling signature is 'u(x)'. Here 'x' is an array_like of 'numpy'.
        The return value is a scalar.

    Returns
    -------
    float
        maximal absolute error at the discretization points
    """
    if not isinstance(hat_u, np.ndarray) or not isinstance(u, callable):
        raise TypeError
    if not isinstance(d, int) or not isinstance(n, int):
        raise TypeError


print(idx([36,23,8,1,1],99))
print(inv_idx(69420,5,99))
