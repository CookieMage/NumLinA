""" Gruppe: 21
programm for solving the Poisson problem

functions
---------
rhs()
    computes the right-hand side vector 'b' for a given function f
idx()
    calculates the number of an equation in the Poisson problem for a given decretization
inv_idx()
    calculates the coordinates of a discretization point for a given equation number of the Poisson
    problem
main()
    Example of code that can be run using the provided functions
"""
import numpy as np

def rhs(d : int, n : int, f : callable):    # pylint: disable=invalid-name
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
        if d < 1 or n < 2
    TypeError
        d, n must be int and f must be a callable function
    """
    if not isinstance(d, int):
        raise TypeError('d must be an int.')
    if not isinstance(n, int):
        raise TypeError('n must be an int.')
    if not callable(f):
        raise TypeError('f must be a callable function.')
    if d < 1:
        raise ValueError('Dimension of d must be >= 1')
    if n < 2:
        raise ValueError('Number of intervals in each dimension n must be >= 2')

    sorted_x_d = []
    for i in range((n-1)**d):
        sorted_x_d.append(np.array(inv_idx(i+1, d, n)))

    array_list = [x/n for x in sorted_x_d]
    vector = [f(x)/(n**2) for x in array_list]

    return np.array(vector)

def idx(nx : list, n : int):    # pylint: disable=invalid-name
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
    
    Raises
    ------
    ValueError
        every element of nx must be >= n
    TypeError
        n must be int and nx must be a list
    """
    if not isinstance(nx, list):
        raise TypeError('nx must be a list.')
    if not isinstance(n, int):
        raise TypeError('n must be an int.')
    for e in nx:    # pylint: disable=invalid-name
        if e >= n:
            raise ValueError(f'Every element of nx must be >= n. The problem was {e} < {n}')
    num = nx[0]
    # Als Alternative wäre es hier nur möglich zu schreiben:
    # for i,e in enumerate(nx[1:],1):
    # Durch das Kopieren der Liste ( durch nx[1:]) ist dies allerdings unnötig aufwendig :(
    for i in range(1,len(nx)):
        num += (n-1)**i * (nx[i]-1)
    return num

def inv_idx(m : int, d : int, n : int): # pylint: disable=invalid-name
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
    
    Raises
    ------
    ValueError
        m must be >(n-1)^d
    TypeError
        d, n, m must be int
    """
    if not isinstance(m, int):
        raise TypeError('m must be an int')
    if not isinstance(d, int):
        raise TypeError('d must be an int')
    if not isinstance(n, int):
        raise TypeError('n must be an int')
    if m > (n-1)**d:
        raise ValueError('m must be > (n-1)^d')
    m -= 1
    nx = [1] * d    # pylint: disable=invalid-name
    # calculate coordinates of the discretization point
    for i in range(len(nx),0,-1):
        nx[i-1] = 1 + (m // ((n-1)**(i-1)))
        m = m % (n-1)**(i-1)
    return nx

def main():
    """ Example of code that can be run using the provided functions
    """
    print(idx([36,23,8,1,1],99))
    print(inv_idx(69420,5,99))

    def f(array : list): # pylint: disable=invalid-name
        return (array[0]*array[1])/(array[1])**2

    print(rhs(d = 2, n = 3, f=f))

if __name__ == "__main__":
    main()
