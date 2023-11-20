def rhs(d, n, f):
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

def idx(nx, n):
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
    if nx[0] > (n-1):
        raise ValueError
    num = nx[0]
    for i in range(1,len(nx)):
        if nx[i] > (n-1):
            raise ValueError
        num = num + (n-1)**i * (nx[i]-1)
             
    return num

def inv_idx(m, d, n):
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
    if m > (n-1)**d:
        raise ValueError
    M = m-1
    nx = [1] * d 
    for i in range(len(nx),0,-1):
        nx[i-1] = nx[i-1] + (M // ((n-1)**(i-1)))
        M = M % (n-1)**(i-1)

    return(nx)

def compute_error(d, n, hat_u, u):
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

print(idx([36,23,8,1,1],99))
print(inv_idx(69420,5,99))