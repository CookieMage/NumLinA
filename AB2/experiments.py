'''Gruppe: 21

program for conduction experiments on block_matrix, linear_solvers and poisson_problem

functions
---------
graph_sparse_dense()
    creates a plot representing the number of non-zero-entries for sparse and non-sparse matrices
    of type BlockMatrix
main()
    Example of code that can be run using the provided class and methods
'''

import time
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import linear_solvers as linsol
from block_matrix import BlockMatrix
import poisson_problem as pp
from plotter import plotter

def graph_sparse_dense(d=1, maximum=5, n=25): #pylint: disable=invalid-name
    '''creates a plot representing the number of non-zero-entries for sparse and non-sparse
    matrices of type BlockMatrix; the data is calculated to reduce runtime and needed resources;
    additional graphs are plotted in order to show the space complexity class

    Parameters
    ----------
    d : int, optional
        the dimension that is represented in the graph, by default 1
    maximum : int, optional
        10**maximum is the maximum number slices for a BlockMatrix, by default 2
    n : int, optional
        describes the number of datapoints for each graph, by default 25
    '''
    # create list for plotting the data
    x_values = np.logspace(0.4, maximum, dtype=int, num=n)
    x_values = [int(x)**d for x in x_values]

    data = [[],[],[],[]]
    # calculate absolute entries, non-zero-entries and reference values
    for x in x_values:  #pylint: disable=invalid-name
        abs_non_zero = (x-1)**d+2*d*(x-2)*(x-1)**(d-1)
        abs_entries = ((x-1)**d)**2

        data[0] += [abs_non_zero]
        data[1] += [abs_entries]
        data[2] += [x**d]
        data[3] += [(x**d)**2]

    # create plot
    _, ax1 = plt.subplots(figsize=(5, 5))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xscale("log")
    plt.yscale("log")
    plt.ylabel("y", fontsize = 20, rotation = 0)
    ax1.yaxis.set_label_coords(-0.01, 1)
    plt.xlabel("N", fontsize = 20)
    ax1.xaxis.set_label_coords(1.01, -0.05)
    ax1.yaxis.get_offset_text().set_fontsize(20)
    ax1.grid()

    # plot data
    plt.plot(x_values, data[0], label = f"sparse matrix d = {d}", linewidth=2, linestyle="dashdot")
    plt.plot(x_values, data[1], label = f"dense matrix  d = {d}", linewidth=2, linestyle="dashdot")
    plt.plot(x_values, data[2], label = f"Referenzwert n^{d}", linewidth=2, linestyle="dotted")
    plt.plot(x_values, data[3], label = f"Referenzwert (n^{d})^2", linewidth=2, linestyle="dotted")

    plt.legend(fontsize=20, loc="upper left")
    plt.show()


def comp_alt_solve_lu(p : np.ndarray, l : np.ndarray, u : np.ndarray, b : np.ndarray):
    '''function for comparing the runtime of linsol.solve_lu() and linsol.solve_lu_alt()

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
    int
        difference in runtime between the two functions (>0 means alternative function is slower)
    '''
    start = time.time()
    linsol.solve_lu(p, l, u, b)
    end = time.time()
    
    start_alt = time.time()
    linsol.solve_lu_alt(p, l, u, b)
    end_alt = time.time()
    
    time_alt = end_alt - start_alt
    time_normal = end - start
    
    return time_alt-time_normal

FAST_MODE=True

def Hallo(d: int , n : int ):
    '''
    ''' 
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
         b_vector.append(pp.pp_zu_bsp_1(x)*(-h**2))

    # use fast mode or not as specified above
    if FAST_MODE:
        loesung = linsol.solve_lu(mat_p, mat_l, mat_u, b_vector)
    else:
        loesung = linsol.solve_lu_alt(mat_p, mat_l, mat_u, b_vector)

    maximum = pp.compute_error(d,n,loesung,pp.bsp_1)
    return maximum





def main():

    N_plot = []
    M_plot = []
    d = 2
    n = np.logspace(0.4, 2.0, 100, dtype=int)
    n = [int(e) for e in n]
    for e in n:
        print(e)
        N = (e-1)**d
        M = Hallo(d,e)
        N_plot.append(N)
        M_plot.append(M)
    print("N:" ,N_plot)
    print("M:" , M_plot)
    print(len(M_plot))
    print(len(N_plot))

    plotter([N_plot],[M_plot], ["Maximalfehler"], ["solid"],["b"])

    

 

    graph_sparse_dense(d=3)

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

    same = 0
    faster = 0
    slower = 0
    # we can statistically test which function is faster most of the time
    for _ in range(1000):
        time_dif = comp_alt_solve_lu(p, l, u, b)
        if time_dif < 0:
            # add 1 if the alternative function ist faster
            faster += 1
        elif time_dif > 0:
            slower += 1
        else:
            same += 1

    print("-------------------------MAIN-START-------------------------")
    text_1 = "\nThe alternative function was "
    print(text_1 + f"faster in this test about {faster/10}% of the time.",
          text_1 + f"slower in this test about {slower/10}% of the time.",
          text_1 + f"as fast as the normal function in this test about {same/10}% of the time.")
    print("\n--------------------------MAIN-END--------------------------")

if __name__ == "__main__":
    main()
