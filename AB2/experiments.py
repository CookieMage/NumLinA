''' Gruppe: 21
programm for experimenting with block_matrix.py and poisson_problem.py

functions
---------
ploter()
    plots provided lists on a graph
graph()
    creates different graphs for comparing the sparse and numpy matrices
main()
    Example of code that can be run using the provided functions
'''
from matplotlib import pyplot as plt
from block_matrix import BlockMatrix
#import poisson_problem

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
    plt.ylabel("Eintraege", fontsize = 20, rotation = 0)
    ax1.yaxis.set_label_coords(-0.01, 1)
    plt.xlabel("N", fontsize = 20)
    ax1.xaxis.set_label_coords(1.01, -0.05)
    ax1.yaxis.get_offset_text().set_fontsize(20)
    ax1.grid()

    # plot data
    plt.plot(x_values[0], plots[0], label = "d = 1 sparse-matrix", linewidth=2, linestyle="solid")
    plt.plot(x_values[0], plots[1], label = "d = 1 full matrix", linewidth=2, linestyle="solid")
    plt.plot(x_values[1], plots[2], label = "d = 2 sparse-matrix", linewidth=2, linestyle="dashdot")
    plt.plot(x_values[1], plots[3], label = "d = 2 full matrix", linewidth=2, linestyle="dashdot")
    plt.plot(x_values[2], plots[4], label = "d = 3 sparse-matrix", linewidth=2, linestyle="dotted")
    plt.plot(x_values[2], plots[5], label = "d = 3 full matrix", linewidth=2, linestyle="dotted")

    plt.legend(fontsize=20, loc="upper left")
    plt.show()

def graph(x_values : list):
    '''creates different graphs for comparing the sparse and numpy matrices

    Parameters
    ----------
    x_values : list
        list of values for the x-axis / experiments

    Returns
    -------
    list of 6
        list containing the experiment-data
    '''
    # create lists for organizing the data 
    sparse_data1 = []
    sparse_data2 = []
    sparse_data3 = []

    full_data1 = []
    full_data2 = []
    full_data3 = []

    data = [sparse_data1, full_data1, sparse_data2, full_data2, sparse_data3, full_data3]

    # experiment on every n for n in x_values
    for n in x_values:
        # create matrices (d= 1, 2, 3)
        mat1 = BlockMatrix(1, n)
        mat2 = BlockMatrix(2, n)
        mat3 = BlockMatrix(3, n)

        # get information of sparsity
        sparse_data1 += [mat1.eval_sparsity()[0]]
        sparse_data2 += [mat2.eval_sparsity()[0]]
        sparse_data3 += [mat3.eval_sparsity()[0]]

        # compute absolute number of entries / number of entries in a full matrix
        full_data1 += [(mat1.n-1)**(2*mat1.d)]
        full_data2 += [(mat2.n-1)**(2*mat2.d)]
        full_data3 += [(mat3.n-1)**(2*mat3.d)]

    return data

def main():
    '''Example of code that can be run using the provided functions
    '''
    x_values = list(range(2, 100, 4))
    
    data = graph(x_values)
    
    x_values = [[x-1 for x in x_values]]
    x_values += [[(x)**2 for x in x_values[0]]]
    x_values += [[(x)**3 for x in x_values[0]]]

    plotter(x_values, data)


if __name__ == "__main__":
    main()
