"""programm for solving the Poisson problem

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
from block_matrix import BlockMatrix
import linear_solvers as linsol
import matplotlib.pyplot as plt

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

    array_list = [((1/n)*x) for x in sorted_x_d]
    vector = [f(x) for x in array_list]

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
    # Durch die Kopierung der Liste (nx[1:]) ist dies unnötige aufwendig.
    for i in range(1,len(nx)):
        num = num + (n-1)**i * (nx[i]-1)
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
    for i in range(len(nx),0,-1):
        nx[i-1] = 1 + (m // ((n-1)**(i-1)))
        m = m % (n-1)**(i-1)
    return nx

def compute_error(d : int, n : int, hat_u : np.ndarray, u : callable):  # pylint: disable=invalid-name
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

    Raises
    ------
    TypeError
        hat_u must be a ndarray
        u must be a callable function
        d and n must be of type int
    """
    if not isinstance(hat_u, np.ndarray):
        raise TypeError('hat_u must be a np.ndarray')
    if not callable(u):
        raise TypeError('u must be a callable function')
    if not isinstance(d, int):
        raise TypeError('d must be an int')
    if not isinstance(n, int):
        raise TypeError('n must be an int')
    block = BlockMatrix(d, n)
    p, l, u = block.get_lu()
    b = rhs(d, n, u)
    solution = linsol.solve_lu(p, l, u, b)
    return max([solution[i]-e for i,e in enumerate(hat_u)])

def plotter(x_values : list, plots : list):
    '''plots provided lists of plots relative to provided list x_values

    Parameters
    ----------
    x_values : list
        list of values for the x-axis
    plots : list
        list of lists of y-values for plots
    '''
    # create the plot
    _, ax1 = plt.subplots(figsize=(5, 5))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xscale("log")
    plt.yscale("log")
    #plt.title(f"{num}. Dimension", fontsize=20)
    plt.ylabel("maximum error", fontsize = 20, rotation = 0)
    ax1.yaxis.set_label_coords(-0.01, 1)
    plt.xlabel("N", fontsize = 20)
    ax1.xaxis.set_label_coords(1.01, -0.05)
    ax1.yaxis.get_offset_text().set_fontsize(20)
    ax1.grid()

    # plot data
    plt.plot(x_values[0], plots[0], label = "d = 1", linewidth=2, linestyle="dashdot")
    plt.plot(x_values[1], plots[1], label = "d = 2", linewidth=2, linestyle="dashdot")
    plt.plot(x_values[2], plots[2], label = "d = 3", linewidth=2, linestyle="dashdot")

    plt.legend(fontsize=20, loc="upper left")
    plt.show()

def graph_error(hat_u, u):
    d = [1, 2, 3]
    n = list(range(2, 100, 4))
    data = []
    for e in d:
        data += [[]]
        for f in n:
            data[d-1] += compute_error(e, f, hat_u, u)
    x_values = [[x-1 for x in n]]
    x_values += [[x**2 for x in x_values]]
    x_values += [[x**3 for x in x_values]]

    plotter(x_values, data)

def bsp_1(x : np.array, k :int):
    """calculates the funktion u(x)_n in examplee 2.2 for a vector of the dimension d"""
    d = len(x)
    y = 1
    for i in range(0,d):
        y = y * x[i] * np.sin(k * np.pi * x[i]) 
    return y 

def pp_zu_bsp_1(x : np.array, k :int):
    z = 0
    for i in range(0,len(x)):
        y = k * np.pi * (2 * np.cos(k * np.pi * x[i])-k * np.pi * x[i] * np.sin(k * np.pi *x[i]))
        pro = 1
        for j in range(0,len(x)):
            if i == j:
                continue
            pro = x[j] * np.sin(k * np.pi * x[j]) * pro
        y = y * pro 
        z = z + y 
    return z

def main():
    """ Example of code that can be run using the provided functions
    """
   # print(idx([36,23,8,1,1],99))
    #print(inv_idx(69420,5,99))

    #f = lambda array: (array[0]*array[1])/array[1]**2 #pylint: disable=unnecessary-lambda-assignment

    #print(rhs(d = 2, n = 3, f=f))
    y= bsp_1([1],1)
    print(y , " BSP 1.--------")
    z = pp_zu_bsp_1([1], 1)
    print(z, "<----- pp_bs1")

    



if __name__ == "__main__":
    main()
