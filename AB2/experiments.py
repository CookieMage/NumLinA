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

import matplotlib.pyplot as plt
import numpy as np

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
    x_values = np.logspace(0.4, maximum, dtype=int, num=n)
    x_values = [int(x)**d for x in x_values]

    data = [[],[],[],[]]
    for x in x_values:  #pylint: disable=invalid-name
        abs_non_zero = (x-1)**d+2*d*(x-2)*(x-1)**(d-1)
        abs_entries = ((x-1)**d)**2

        data[0] += [abs_non_zero]
        data[1] += [abs_entries]
        data[2] += [x**d]
        data[3] += [(x**d)**2]

    _, ax1 = plt.subplots(figsize=(5, 5))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xscale("log")
    plt.yscale("log")

    plt.ylabel("Eintraege", fontsize = 20, rotation = 0)
    ax1.yaxis.set_label_coords(-0.01, 1)
    plt.xlabel("N", fontsize = 20)
    ax1.xaxis.set_label_coords(1.01, -0.05)
    ax1.yaxis.get_offset_text().set_fontsize(20)
    ax1.grid()

    plt.plot(x_values, data[0], label = f"sparse matrix d = {d}", linewidth=2, linestyle="dashdot")
    plt.plot(x_values, data[1], label = f"dense matrix  d = {d}", linewidth=2, linestyle="dashdot")
    plt.plot(x_values, data[2], label = f"Referenzwert n^{d}", linewidth=2, linestyle="dotted")
    plt.plot(x_values, data[3], label = f"Referenzwert (n^{d})^2", linewidth=2, linestyle="dotted")

    plt.legend(fontsize=20, loc="upper left")
    plt.show()

def main():
    '''Example of code that can be run using the provided class and methods
    '''
    graph_sparse_dense(d=3)

if __name__ == "__main__":
    main()
