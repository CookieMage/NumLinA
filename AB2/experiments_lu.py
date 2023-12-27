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
import linear_solvers as linsol
import block_matrix
import poisson_problem as pp

FAST_MODE = True

def graph_approx_anal(pp_u : callable, u : callable, n=20, d = 1):  #pylint: disable=(invalid-name, disable=too-many-locals
    '''creates a plot representing the approximative and analytic solution of the Poisson-problem

    Parameters
    ----------
    pp_u : callable
        functions within the Poisson-problem
    u : callable
        analyitic solution of the Poisson-problem
    n : int, optional
        number of intervals of approximation, by default 20
    d : int, optional
        dimension of the space, by default 1
    '''
    h = 1/n #pylint: disable=invalid-name
    # create coefficient matrix A for given n and d
    mat = block_matrix.BlockMatrix(d,n)
    mat_p, mat_l, mat_u = mat.get_lu()  #pylint: disable=invalid-name, disable=unbalanced-tuple-unpacking

    #b_vector = pp.rhs(n,d,pp.pp_zu_bsp_1)
    b_vector = []
    disc_points = []
    for i in range(1, (n-1)**d+1):
        # create list of discretization points
        x = pp.inv_idx(i,d,n)  #pylint: disable=invalid-name
        x = [j/n for j in x]    #pylint: disable=invalid-name
        #calculate right side of f(x)*(-h^2)=b
        b_vector.append(pp_u(x)*(-h**2))
        disc_points.append(x)

    # use fast mode or not as specified above
    if FAST_MODE:
        loesung = linsol.solve_lu(mat_p, mat_l, mat_u, b_vector)
    else:
        loesung = linsol.solve_lu_alt(mat_p, mat_l, mat_u, b_vector)

    labels = ["numerische Loesung", "analytische Loesung"]
    linestyles = ["solid", "dashdot"]
    colors = ["b", "r"]

    # create the plot
    _, ax1 = plt.subplots(figsize=(5, 5))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.ylabel("y", fontsize = 20, rotation = 0)
    ax1.yaxis.set_label_coords(-0.01, 1)
    plt.xlabel("x", fontsize = 20)
    ax1.xaxis.set_label_coords(1.01, -0.05)
    ax1.yaxis.get_offset_text().set_fontsize(20)
    ax1.grid()

    for i,e in enumerate([loesung, [u(x) for x in disc_points]]):    #pylint: disable=invalid-name
        plt.plot(disc_points, e, label = labels[i], linewidth=2, linestyle=linestyles[i],
                 color=colors[i])

    plt.legend(fontsize=20)
    plt.show()


def graph_sparse_dense(d=1, maximum=5, num=25): #pylint: disable=invalid-name
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
    x_values = np.logspace(0.4, maximum, dtype=int, num=num)
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


def comp_alt_solve_lu(p : np.ndarray, l : np.ndarray, u : np.ndarray, b : np.ndarray):  #pylint: disable=invalid-name
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


def main():
    '''Example of code that can be run using the provided class and methods
    '''
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
    for _ in range(10000):
        time_dif = comp_alt_solve_lu(p, l, u, b)
        if time_dif < 0:
            # add 1 if the alternative function ist faster
            faster += 1
        elif time_dif > 0:
            slower += 1
        else:
            same += 1

    print("\n-------------------------MAIN-START-------------------------\n")
    dim = None
    print("Es wird nun der Speicherplatzbedarf von sparse und dense Matrizen verglichen.")
    while not isinstance(dim, int):
        dim = input("Fuer welche Dimension soll die sparsity geplottet werden? (0<d<4)\n")
        try:
            dim=int(dim)
            if 1>dim or 3<dim:
                dim = "Mach ich nicht"
        except ValueError:
            continue
    graph_sparse_dense(d=dim)

    print("\nEs folgt die Graphik der Approximation des Poisson Problems und die analytische",
          "Loesung von diesem.")
    input("Bitte bestaetigen Sie dies mit ENTER.")
    graph_approx_anal(pp.pp_zu_bsp_1, pp.bsp_1)

    print("\nEs folgt die Graphik, welche die Nicht-Null-Eintraege der LU-Zerlegung",
          "und der Matrix selbst vergleicht.")
    input("Bitte bestaetigen Sie dies mit ENTER.")
    block_matrix.graph_lu()

    print("\nEs folgen die sparse-vs-dense-Graphiken fuer verschieden Dimensionen.")
    input("Bitte bestaetigen Sie mit dies ENTER.")
    block_matrix.graph_sparse_dense(dim=[1])
    block_matrix.graph_sparse_dense(dim=[2])
    block_matrix.graph_sparse_dense(dim=[3])
    block_matrix.graph_sparse_dense(dim=[1,2,3])

    print("\nNun wird unsere alternative Implementation von solve_lu() mit der ",
          "Standardimplementation verglichen:")
    text_0 = "Die alternative Funktion war in diesem Test "
    text_1 = text_0+f"schneller in {faster/100}% der Versuche.\n"
    text_1 += text_0+f"langsamer in {slower/100}% of der Versuche.\n"
    text_1 += text_0+f"gleichschnell wie die Standardimplementation in {same/100}% der Versuche.\n"
    print(text_1)
    print("\n--------------------------MAIN-END--------------------------\n")

if __name__ == "__main__":
    main()
