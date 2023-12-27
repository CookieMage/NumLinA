'''Gruppe: 21

program for supporting block_matrix and poisson_problem in regards of creating plots

functions
---------
plotter()
    plots a list of numbers onto another list of numbers using specific labels, etc.
main()
    Example of code that can be run using the provided functions
'''
import matplotlib.pyplot as plt

def plotter(x_values : list, plots : list, labels : list, linestyles : list, colors : list):
    '''plots provided lists of plots relative to provided list x_values

    Parameters
    ----------
    x_values : list of lists of int or float
        list of lists of values for the x-axis
    plots : list of lists of int or float
        list of lists of y-values for plots
    labels : list of str
        list of labels for the different graphs
    linestyles : list of str
        list of linestyles for the different graphs
    colors : list of str
        list of colors for the different graphs
    '''
    # create the plot
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

    # plot datasets
    for i,e in enumerate(plots):    #pylint: disable=invalid-name
        plt.plot(x_values[0], e, label = labels[i], linewidth=2, linestyle=linestyles[i],
                 color=colors[i])

    plt.legend(fontsize=20, loc="upper left")
    plt.show()

def main():
    '''Example of code that can be run using the provided class and methods
    '''
    x_values = list(range(1, 100))
    y_values = [[x for x in x_values]]
    y_values += [[x**2 for x in x_values]]
    y_values += [[1/x for x in x_values]]
    x_values = [x_values]*3
    labels = ["f(x)=x", "f(x)=x^2", "f(x)=1/x"]
    linestyles = ["dotted", "dashed", "dashdot"]
    colors = ["b", "r", "c"]
    print("\n-------------------------MAIN-START-------------------------\n")
    print(f"Dies ist eine mittels plotter() erstellte Graphik, welche {labels} darstellt.")
    plotter(x_values, y_values, labels, linestyles, colors)
    print("\n--------------------------MAIN-END--------------------------\n")

if __name__ == "__main__":
    main()
