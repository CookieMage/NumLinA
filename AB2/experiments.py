import matplotlib.pyplot as plt
import linear_solvers as linsol
import poisson_problem as pp
from block_matrix import BlockMatrix
import numpy as np

def test_1(max_n, slices_n, max_d, slices_d):
    n_list = np.logspace(2, max_n, slices_n, dtype=int)
    d_list = np.logspace(1, max_d, slices_d, dtype=int)
    
    hat_u = pp.bsp_1
    u = pp.pp_zu_bsp_1
    
    for n in n_list:
        for d in d_list:
            errors += pp.compute_error(d, n, hat_u, u)


def graph_sparse_dense(d=1, x=5, n=25):
    x_values = np.logspace(0.4, x, dtype=int, num=n)
    x_values = [int(x)**d for x in x_values]
    
    data = [[],[],[],[]]
    for x in x_values:
        #print(n)
        # create matrices (d= 1, 2, 3)
        abs_non_zero = (x-1)**d+2*d*(x-2)*(x-1)**(d-1)
        abs_entries = ((x-1)**d)**2


        # get information of sparsity
        data[0] += [abs_non_zero]
        data[1] += [abs_entries]
        data[2] += [x**d]
        data[3] += [(x**d)**2]
    # create the plot
    _, ax1 = plt.subplots(figsize=(5, 5))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xscale("log")
    plt.yscale("log")
    #plt.title(f"{num}. Dimension", fontsize=20)
    plt.ylabel("Eintraege", fontsize = 20, rotation = 0)
    ax1.yaxis.set_label_coords(-0.01, 1)
    plt.xlabel("N", fontsize = 20)
    ax1.xaxis.set_label_coords(1.01, -0.05)
    ax1.yaxis.get_offset_text().set_fontsize(20)
    ax1.grid()

    # plot data
    plt.plot(x_values, data[0], label = f"sparse matrix d = {d}", linewidth=2, linestyle="dashdot")
    plt.plot(x_values, data[1], label = f"dense matrix  d = {d}", linewidth=2, linestyle="dashdot")
    plt.plot(x_values, data[2], label = f"O(n)=n^{d}", linewidth=2, linestyle="dotted")
    plt.plot(x_values, data[3], label = f"O(n)=(n^{d})^2", linewidth=2, linestyle="dotted")
    

    plt.legend(fontsize=20, loc="upper left")
    plt.show()

def main():
    graph_sparse_dense(d=1)
    graph_sparse_dense(d=2)
    graph_sparse_dense(d=3)

if __name__ == "__main__":
    main()