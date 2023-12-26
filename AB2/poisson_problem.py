"""Gruppe: 21

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
from block_matrix import BlockMatrix
import linear_solvers as linsol
from plotter import plotter

# this programm has a fast mode which uses predefined functions of scipy to solve the linear system
# in compute_error() and a slow mode which uses our own function for calculating the solution
FAST_MODE = True

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
    vector = [f(x)/n**2 for x in array_list]

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
    # Als Alternative waere es hier nur moeglich zu schreiben:
    # for i,e in enumerate(nx[1:],1):
    # Durch die Kopierung der Liste (nx[1:]) ist dies unnoetige aufwendig.
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
    h = 1/n #pylint: disable=invalid-name
    values_of_b_vecotor = []  #erstellt Vektor fuer rechte Seite der gleichung
    for i in range(1, (n-1)**d+1):
        #erzeugt eine liste mit den Disrkretisierungspunkten * n
        x = inv_idx(i,d,n)  #pylint: disable=invalid-name
        #bereitet die Diskretisierungspunkte fuer das einsetzen in die funktion vor
        x = [j/n for j in x]    #pylint: disable=invalid-name
        values_of_b_vecotor.append(u(x)*(-h**2))   #setzt die Diskretisierungspunkte in eine
        # funktion f ein (rechte seite)
                                                    #und rechnet f*((-h)^2) statt A * (-1/h^2)
    mat = BlockMatrix(d,n)          #erzeugt die koeffizientenmatrix A zu gegebenen n und d
    p, l, u = mat.get_lu()  #pylint: disable=invalid-name, disable=unbalanced-tuple-unpacking

    if FAST_MODE:  #(True or False) wahlz zwischen loesung durch pyscy oder selbst programierte
        loesung = linsol.solve_lu(p, l, u, values_of_b_vecotor) #loest das lineare gleichungssystem
        # mit pyscy
    else:
        loesung = linsol.solve_lu_alt(p, l, u, values_of_b_vecotor) #loest das LGS mit eigener
        # Funktion

    maximum = max(abs(e-hat_u[i]) for i,e in enumerate(loesung))  #pylint: disable=invalid-name
    return maximum


def graph_error(u : callable, pp_u : callable):   #pylint: disable=invalid-name
    '''graphs the error of the numerical solution of the Poisson problem
    with respect to the infinity-norm

    Parameters
    ----------
    u : callable
        function that is used for the numerical solution of the Poisson problem
    pp_u : callable
        analytic solution of the Poisson problem
    '''
    dim = [1, 2, 3]
    n = np.logspace(0.4, 1.4, 5, dtype=int) #pylint: disable=invalid-name
    n = [int(e) for e in n] #pylint: disable=invalid-name
    data = []

    for d in dim:   #pylint: disable=invalid-name
        data += [[]]
        for e in n: #pylint: disable=invalid-name
            block = BlockMatrix(d, e)
            p_mat, l_mat, u_mat = block.get_lu()    #pylint: disable=unbalanced-tuple-unpacking
            disc_points = [inv_idx(m, d, e) for m in range(1, (e-1)**d+1)]
            disc_points = [[x/e for x in y] for y in disc_points]
            solutions = np.append([], linsol.solve_lu(p_mat, l_mat, u_mat,
                                                      [u(x) for x in disc_points]))
            data[d-1] = np.append(data[d-1], compute_error(d=d, n=e, hat_u=np.array(solutions),
                                                           u=pp_u))

    x_values = n
    x_values = [[int(x)**3 for x in x_values],
                [int(int(x)**1.5) for x in x_values],
                [int(x) for x in x_values]]

    labels = [f"error d={d}" for d in dim]
    linestyles = ["dashdot"]*3
    colors = ["b", "r", "c"]

    plotter(x_values, data, labels, linestyles, colors)



def bsp_1(x : np.array, k=1): #pylint: disable=missing-function-docstring, disable=invalid-name
    y = 1   #pylint: disable=invalid-name
    for e in x: #pylint: disable=invalid-name
        y = y * e * np.sin(k * np.pi * e)   #pylint: disable=invalid-name
    return y

def pp_zu_bsp_1(x : np.array, k=1): #pylint: disable=missing-function-docstring, disable=invalid-name
    z = 0   #pylint: disable=invalid-name
    for i,e in enumerate(x):    #pylint: disable=invalid-name
        y = k * np.pi * (2 * np.cos(k * np.pi * e)-k * np.pi * e * np.sin(k * np.pi *e))    #pylint: disable=invalid-name
        pro = 1
        for j,f in enumerate(x):    #pylint: disable=invalid-name
            if i != j:
                pro *= f * np.sin(k * np.pi * f)
        y *= pro    #pylint: disable=invalid-name
        z += y  #pylint: disable=invalid-name
    return z



def main():
    """ Example of code that can be run using the provided functions
    """
    graph_error(bsp_1, pp_zu_bsp_1)
   # print(idx([36,23,8,1,1],99))
    #print(inv_idx(69420,5,99))

    #f = lambda array: (array[0]*array[1])/array[1]**2 #pylint: disable=unnecessary-lambda-assignment

    #print(rhs(d = 2, n = 3, f=f))
    #y= bsp_1([1],1)
    #print(y , " BSP 1.--------")
    #z = pp_zu_bsp_1([1], 1)
    #print(z, "<----- pp_bs1")
    #n= 20
    #d = 2
    #values_of_u_vector = []
    #for i in range(1,1+(n-1)**d):
    #    x = inv_idx(i,d,n)
    #    x = [j/n for j in x]
    #    values_of_u_vector.append(bsp_1(x))



if __name__ == "__main__":
    main()
