'''This is an implementation of the class FiniteDifference, which can be used to approximate the
first and second derivatives of a given function. It can be used to find the error of these
approximations as well.

Classes
-------
FiniteDifference
    Represents the first and second order finite difference approximation of a function and
    allows for a computation of error to the exact derivatives.

Functions
---------
power_func()
    function that draws the identity, square and cube functions as well as the first and second
    derivative of these for given values x
plot_derivatives()
    function for drawing the plots of the approximated derivatives of func_f and its analytic
    derivatives if these have been provided
plothelper()
    support function for main()
plot()
    support function for main()
progress_bar()
    function to support main(); displays a progressbar in console
main()
    example implementation for using the above defined code
'''
import math
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import manage_list as ml



class FiniteDifference:
    ''' Represents the first and second order finite difference approximation of a function and
    allows for a computation of error to the exact derivatives.

    Parameters
    ----------
    h : float
        Step size of the approximation.
    f : callable
        Function to approximate the derivatives of. The calling signature is
        'f(x)'. Here 'x' is a scalar or array_like of 'numpy'. The return
        value is of the same type as 'x'.
    d_f : callable, optional
        The analytic first derivative of 'f' with the same signature.
    dd_f : callable, optional
        The analytic second derivative of 'f' with the same signature.

    Attributes
    ----------
    h : float
        Step size of the approximation.
    f : callable
        Function to approximate the derivatives of. The calling signature is
        'f(x)'. Here 'x' is a scalar or array_like of 'numpy'. The return
        value is of the same type as 'x'.
    d_f : callable, default value = None
        The analytic first derivative of 'f' with the same signature.
    dd_f : callable, default value = None
        The analytic second derivative of 'f' with the same signature.
    
    Functions
    ---------
    __init__()
        returns an Object of class FiniteDifference
    compute_dhs_f(self):
        use a alternative method to calculates the approximation for the first derivative of the f
        with step size h using a better approximation of O(h^4)
    compute_dh_f()
        calculates the approximation for the first derivative of the f with step size h
    compute_ddh_f()
        calculates the approximation for the second derivative of f with step size h
    compute_errors()
        Calculates an approximation to the errors between an approximation
        and the exact derivative for first and second order derivatives in the
        infinity norm.
    compute_dhs_errors()
        Calculates an approximation to the errors between an approximation
        and the exact derivative for first order derivatives in the
        infinity norm.
    experiment()
        approximates first and second deriviatves, computes their values on a given Intervall
        and saves them; calculates the error of these approximations and saves them as well
    '''
    h = None        # pylint: disable=invalid-name
    f = None        # pylint: disable=invalid-name
    d_f = None
    dd_f = None

    def __init__(self, h : float, f : callable, d_f=None, dd_f=None):   # pylint: disable=invalid-name
        '''returns an Object of class FiniteDifference

        Parameters
        ----------
        h: float
            Step size of the approximation.
        f : callable
            Function to approximate the derivatives of. The calling signature is
            'f(x)'. Here 'x' is a scalar or array_like of 'numpy'. The return
            value is of the same type as 'x'.
        d_f : callable, optional
            The analytic first derivative of 'f' with the same signature.
        dd_f : callable, optional
            The analytic second derivative of 'f' with the same signature.

        Raises
        ------
        TypeError
            One of the given arguments is of the wrong type
        '''
        # raise error if parameters are of the wrong type
        # assign values of parameters to attributes of object
        try:
            h = float(h)
        except TypeError as exc:
            raise exc
        if not callable(f):
            raise TypeError
        if d_f is not None and not callable(d_f):
            raise TypeError
        if dd_f is not None and not callable(dd_f):
            raise TypeError
        self.h = h          # pylint: disable=invalid-name
        self.f = f          # pylint: disable=invalid-name
        self.d_f = d_f
        self.dd_f = dd_f

    def compute_dhs_f(self):
        '''use a alternativ method to calculates the approximation for the first derivative of the
        f with step size h with a better approximation of O(h^4)

        Parameters
        ----------

        Return
        ------
        callable
            calculates the approximation of the first derivative for a given x
        '''
        # define f_1 as the approximation of the first derivative of f using step size h
        def fs_1(x_value : float or int):
            return (self.f(x_value-2* self.h)-8 * self.f(x_value-self.h)+8 * \
                self.f(x_value+self.h)-self.f(x_value+2 * self.h)) / (12*self.h)
        return fs_1

    def compute_dh_f(self):
        '''calculates the approximation for the first derivative of the f with step size h whith
        the forrword difference

        Parameters
        ----------

        Return
        ------
        callable
            calculates the approximation of the first derivative for a given x
        '''
        # define f_1 as the approximation of the first derivative of f using step size h
        def f_1(x_value : float or int):
            return (self.f(x_value+self.h)-self.f(x_value))/self.h
        return f_1

    def compute_ddh_f(self):
        '''calculates the approximation for the second derivative of f with step size h

        Parameters
        ----------

        Return
        ------
        callable
            calculates the approximation of the second derivative for a given x
        '''
        # define f_2 as the approximation of the second derivative of f using step size h
        def f_2(x_value : float or int):
            return (self.f(x_value+self.h)-2*self.f(x_value)+self.f(x_value-self.h))/(self.h**2)
        return f_2

    def compute_errors(self, a : float, b : float, p : int):    # pylint: disable=invalid-name
        ''' Calculates an approximation to the errors between an approximation
        and the exact derivative for first and second order derivatives in the
        infinity norm.
        
        Parameters
        ----------
        a, b : float
            Start and end point of the interval.
        p : int
            Number of intervals used in the approximation of the infinity norm.

        Returns
        -------
        float
            Errors of the approximation of the first derivative.
        float
            Errors of the approximation of the second derivative.

        Raises
        ------
        ValueError
            If no analytic derivative has been provided
        '''
        # raise ValueError if no analytic derivative was provided
        if self.d_f is None and self.dd_f is None:
            raise ValueError
        # compute approximations of first and second derivatives
        f_1 = self.compute_dh_f()
        f_2 = self.compute_ddh_f()
        # declaration and initialization of max_dif_1, max_dif_2
        max_dif_1 = 0.0
        max_dif_2 = 0.0
        # create list containing p evenly spaced numbers within the interval [a,b]
        steps = np.linspace(a, b, p)
        # find the biggest local difference between the analytic and approximated first derivative
        # if an analytic one was provided
        if self.d_f is not None:
            for i in steps:
                max_dif_1 = max(max_dif_1, abs(f_1(i) - self.d_f(i)))
        # find the biggest local difference between the analytic and approximated second derivative
        # if an analytic one was provided
        if self.d_f is not None:
            for i in steps:
                max_dif_2 = max(max_dif_2, abs(f_2(i) - self.dd_f(i)))
        return max_dif_1, max_dif_2


    def compute_dhs_errors(self, a : float, b : float, p : int):    # pylint: disable=invalid-name
        ''' Calculates an approximation to the errors between an approximation
        and the exact derivative for first order derivatives in the
        infinity norm.
        
        Parameters
        ----------
        a, b : float
            Start and end point of the interval.
        p : int
            Number of intervals used in the approximation of the infinity norm.

        Returns
        -------
        float
            Errors of the approximation of the first derivative.
        float
            Errors of the approximation of the second derivative.

        Raises
        ------
        ValueError
            If no analytic derivative has been provided
        '''
        # raise ValueError if no analytic derivative was provided
        if self.d_f is None:
            raise ValueError
        # compute approximations of first
        f_1 = self.compute_dhs_f()
        # declaration and initialization of max_dif_1, max_dif_2
        max_dif_1 = 0.0
        # create list containing p evenly spaced numbers within the interval [a,b]
        steps = np.linspace(a, b, p)
        # find the biggest local difference between the analytic and approximated first derivative
        for i in steps:
            max_dif_1 = max(max_dif_1, abs(f_1(i) - self.d_f(i)))
        return max_dif_1


    def experiment(self, func_name : str, a : float, b : float, p : int):   # pylint: disable=invalid-name
        '''approximates first and second deriviatves, computes their values on a given Intervall
        and saves them; calculates the error of these approximations and saves them as well

        Parameters
        ----------
        func_name : str
            name of function that is used for experimentation
            only used for labeling the save data
        a, b : float
            Start and end point of the interval.
        p : int
            Number of intervals used in the approximation of the infinity norm.
        
        Raises
        ------
        ValueError
            If no analytic derivative has been provided
        '''
        # create list containing p evenly spaced numbers within the interval [a,b]
        steps = np.linspace(a, b, p)
        # compute approximations of the first and second derivatives
        dh_f = self.compute_dh_f()
        ddh_f = self.compute_ddh_f()
        dhs_f = self.compute_dhs_f()
        # compute the maximum local difference between the analytic and approximated derivatives
        errors = self.compute_errors(a, b, p)
        dhs_error = self.compute_dhs_errors(a, b, p)
        # save error data
        ml.save(func_name, "_error.csv", "dh_1", self.h, errors[0])
        ml.save(func_name, "_error.csv", "ddh_2", self.h, errors[1])
        ml.save(func_name, "_dhs_error.csv", "dhs", self.h, dhs_error)
        # save an empty line in the error file in order to make different experiments done in
        # succesion distinguishable
        ml.save(func_name, "_error.csv", "")
        ml.save(func_name, "_dhs_error.csv", "")
        # save function values for each values in steps in corresponding files
        for i in steps:
            ml.save(func_name, "_dh.csv", i, dh_f(i))
            ml.save(func_name, "_ddh.csv", i, ddh_f(i))
            ml.save(func_name, "_1.csv", i, self.d_f(i))
            ml.save(func_name, "_2.csv", i, self.dd_f(i))
            ml.save(func_name, "_dhs.csv", i, dhs_f(i))
        # save an empty line in the corresponding file in order to make different experiments done
        # in succesion distinguishable
        ml.save(func_name, ".csv", "")
        ml.save(func_name, "_dh.csv", "")
        ml.save(func_name, "_ddh.csv", "")
        ml.save(func_name, "_1.csv", "")
        ml.save(func_name, "_2.csv", "")
        ml.save(func_name, "_dhs.csv", "")



def power_func(numbers : list):
    '''function that draws the identity, square and cube functions as well as the first and second
    derivative of these for given values

    Parameters
    ----------
    numbers : list
        numbers which are used to calculate values of the functions which are then plotted
    '''
    # create list to make labeling easier
    colors = ["b", "g", "r"]
    line = ["solid", "dashed", "dotted"]
    func_name = ["f", "g", "h"]
    # initialize graph
    _, ax1 = plt.subplots(figsize=(5, 5))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.ylabel("y", fontsize = 20, rotation = 0)
    ax1.yaxis.set_label_coords(-0.01, 1)
    plt.xlabel("x", fontsize = 20)
    ax1.xaxis.set_label_coords(1.02, 0.025)
    ax1.yaxis.get_offset_text().set_fontsize(20)
    plt.yscale("log")
    plt.xscale("log")
    # repeat everything for each s
    for s_value in [1, 2, 3]:
        labels = [f"{func_name[s_value-1]}(x)=x^{s_value}", f"{func_name[s_value-1]}(x)'",
                  f"{func_name[s_value-1]}(x)''"]
        # define the function as well as its first and second analytic derivatives
        def func(x_value):
            return x_value**s_value
        def func_1(x_value):
            return s_value*x_value**(s_value-1)
        def func_2(x_value):
            return (s_value-1)*s_value*x_value**(s_value-2)
        # plot graphs using corresponding colors, linestyles and labels
        for i, element in enumerate([func, func_1, func_2]):
            plt.plot(numbers, [element(x) for x in numbers], label = labels[i],
                     color = colors[s_value-1], linewidth = 2, linestyle = line[i])
    plt.grid()
    plt.legend(fontsize = 20)
    plt.show()


def plot_derivatives(h : int, a : float, b : float, p : int,    # pylint: disable=invalid-name
                     func_f : callable, func_d_f= None,         # pylint: disable=invalid-name
                     func_dd_f = None):                         # pylint: disable=invalid-name
    '''function for drawing the plots of the approximated derivatives of func_f and its analytic
    derivatives if these have been provided
    
    Parameter
    ---------
    h : int
        current number
    a, b : float
        Start and end point of the interval.
    p : int
        Number of intervals used in the approximation of the infinity norm.
    func_f : callable
        function which derivatives are displayed
    func_d_f : callable, optional
        analytic first derivative of func_f
    func_dd_f : callable, optional
        analytic second derivative of func_f
    '''
    steps = np.linspace(a, b, p)
    # compute approximations of the first and second derivatives
    exp = FiniteDifference(h, func_f, func_d_f, func_dd_f)
    dh_f = exp.compute_dh_f()
    ddh_f = exp.compute_ddh_f()
    # initialize graph
    _, ax1 = plt.subplots(figsize=(5, 5))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.ylabel("y", fontsize = 20, rotation = 0)
    ax1.yaxis.set_label_coords(-0.01, 1)
    plt.xlabel("x", fontsize = 20)
    ax1.xaxis.set_label_coords(1.02, 0.025)
    ax1.yaxis.get_offset_text().set_fontsize(20)
    plt.title("Ableitungen", fontsize=40)
    ax1.grid()
    # plot approximation of first derivative
    plt.plot(steps, [dh_f(step) for step in steps], "b",
             label = "1. approximierte Ableitung", linewidth = 2,linestyle = "solid")
    # plot approximation of second derivative
    plt.plot(steps, [ddh_f(step) for step in steps], "g",
             label = "2. approximierte Ableitung", linewidth = 2,linestyle = "solid")
    # plot first derivative if it has been provided
    if func_d_f is not None:
        plt.plot(steps, [func_d_f(step) for step in steps], "b",
                 label = "1. analytische Ableitung", linewidth = 6,linestyle = "dotted")
    # plot second derivative if it has been provided
    if func_dd_f is not None:
        plt.plot(steps, [func_dd_f(step) for step in steps], "g",
                 label = "1. analytische Ableitung", linewidth = 6, linestyle = "dotted")
    plt.legend(fontsize=20, loc="lower left")
    plt.show()



def plothelper(filename : str):
    '''function for reading files created by experiment() and returning the saved data in a usable
    format

    Parameters
    ----------
    filename : str
        name of the file that is to be read

    Returns
    -------
    dh : list
        read data
    ddh : list
        read data
    h : list, None
        read data
    '''
    # open specified file
    with open("experiments/" + filename, "r", encoding = "utf8") as file:
        # read file line per line
        lines = file.readlines()
        # split lines at spaces
        lines = [i.split() for i in lines]
        # initialize and declare lines_list
        lines_list = []

        # if the file does not contain error data open it and read its content
        if "error" not in filename:
            # initialize and declare counter
            # counter is used to dynamicly scale the size of lines_list accordingly
            counter = 0
            # save elements of lines in lines_list, skipping empty str and grouping different
            # paragraphs by saving them as a list in lines_list
            for i,element in enumerate(lines):
                # add a new empty list to lines_list, if lines_list has the same size as counter
                if len(lines_list) == counter:
                    lines_list += [[]]
                # if the element is empty increment counter
                if element == []:
                    counter += 1
                    continue
                # save element in lines_list
                lines_list[counter] += element
            # initialize and declare x, y
            x_values, y_values = [], []
            # take every element of element of lines_list with an even number and append it to
            # x_values
            for element in lines_list:
                x_values += [[element[2*j] for j in range(len(element)//2)]]
            # take every element of element of lines_list with an odd number and append it to
            # y_values
            for element in lines_list:
                y_values += [[element[2*j+1] for j in range(len(element)//2)]]
            # convert all data into floats
            for i,element in enumerate(y_values):
                for j,felement in enumerate(element):
                    y_values[i][j] = float(felement)
            # every element of x is the same so we don't need more than one
            x_values = x_values[0]
            # x ist still a list and every element of that list must be converted to float
            for i,element in enumerate(x_values):
                x_values[i] = float(element)
        # if the file does contain error data open it and read its content
        else:
            # initialize and declare counter
            # counter is used to dynamicly scale the size of lines_list accordingly
            counter = 0
            # save elements of lines in lines_list, skipping empty str and grouping different
            # paragraphs by saving them as a list in lines_list
            for i,element in enumerate(lines):
                # add a new empty list to lines_list, if lines_list has the same size as counter
                if len(lines_list) == counter:
                    lines_list += [[]]
                # if the element is empty increment counter
                if element == []:
                    counter += 1
                    continue
                # save element in lines_list
                lines_list[counter] += element
            # delete str "dh" and "ddh" from lines_list
            for i, element in enumerate(lines_list):
                for j, felement in enumerate(element):
                    if "dh" in felement:
                        lines_list[i].pop(j)        # pylint: disable=unnecessary-list-index-lookup
            # initialize and declar h_values, temp, dh_values, ddh_values
            h_values, temp, dh_values, ddh_values = [], [], [], []
            # save every even element of element of lines_list in h_values
            for element in lines_list:
                h_values += [[element[2*j] for j in range(len(element)//2)]]
            # since dh_values and ddh_values use the same h every element of h looks like [x, x]
            # where x is the value for h_values and therefore we only need the first element of
            # each element of h_values in h_values
            h_values = [element[0] for element in h_values]
            # save every odd element of element of lines_list in temp
            for element in lines_list:
                temp += [[element[2*j+1] for j in range(len(element)//2)]]
            # return formated date if the file contains dhs data
            if "dhs" in filename:
                # remove braces of list within temp ( [x, [y, z]] -> [x, y, z] )
                temp = [inner for outer in temp for inner in outer]
                # convert every element of dhs_values to float
                for i, element in enumerate(temp):
                    temp[i] = float(element)
                return h_values, temp, None
            # save every even element of element of temp in dh_values
            for element in temp:
                dh_values += [[element[2*j] for j in range(len(element)//2)]]
            # save every odd element of element of temp in ddh_values
            for element in temp:
                ddh_values += [[element[2*j+1] for j in range(len(element)//2)]]
            # remove braces of list within dh_values and ddh_values ( [x, [y, z]] -> [x, y, z] )
            dh_values = [inner for outer in dh_values for inner in outer]
            ddh_values = [inner for outer in ddh_values for inner in outer]
            # convert every element of dh_values to float
            for i, element in enumerate(dh_values):
                dh_values[i] = float(element)
            # convert every element of ddh_values in float
            for i, element in enumerate(ddh_values):
                ddh_values[i] = float(element)
            return h_values, dh_values, ddh_values
        return x_values, y_values, None



def plot(title : str, filename_1 : str, name_1 : str, filename_2 = None, name_2 = None):
    '''function for creating a plot containing the data of given files

    Parameters
    ----------
    title : str
        title of the generated plot
    filename_1 : str
        name of the first file containing data
    name_1 : _str
        name of the graph corresponding to the first file
    filename_2 : str, default value = None
        name of the second file containing data
    name_2 : str, default value = None
        name of the graph corresponding to the second file
    '''
    try:
        # initialize graph
        _, ax1 = plt.subplots(figsize=(5, 5))
        plt.xticks(fontsize=17)
        plt.yticks(fontsize=17)
        plt.ylabel("y", fontsize = 20, rotation = 0)
        ax1.yaxis.set_label_coords(-0.01, 1)
        plt.xlabel("x", fontsize = 20)
        ax1.xaxis.set_label_coords(1.02, 0.025)
        ax1.yaxis.get_offset_text().set_fontsize(20)
        plt.title(title, fontsize=40)
        ax1.grid()
        # initialize and declare legend, graph which are later used to create the legend
        legend = []
        graphs = []

        # if the first file does not contain error data continue as usual
        if "error" not in filename_1:
            # using plothelper() read the data for the first graph
            x_values, y1_values, _ = plothelper(filename_1)
            # draw all plots saved in y1_values
            for element in y1_values:
                plt.plot(x_values, element, "g", linewidth = 2, linestyle="solid")
            # append graph to legend
            legend.append(name_1)
            graphs.append(Line2D([0], [0], color = "g", linewidth=2, linestyle="solid"))

            # repeat for the second file if it has been provided
            if filename_2 is not None:
                # using plothelper() read the data for the second graph
                x_values, y2_values, _ = plothelper(filename_2)
                # draw all plots saved in y2_values
                for element in y2_values:
                    plt.plot(x_values, element, "b", linewidth = 4, linestyle="dotted")
                # append graph to legend
                legend.append(name_2)
                graphs.append(Line2D([0], [0], color = "b", linewidth=4, linestyle="dotted"))
        # if the first file does contain error data open it and read its content
        elif "dhs" not in filename_1:
            ax1.set_ylim(bottom = 10**-8.5)
            # change graph accordingly
            plt.ylabel("y", fontsize = 20, rotation = 0)
            ax1.yaxis.set_label_coords(-0.01, 1)
            plt.xlabel("h", fontsize = 20)
            ax1.xaxis.set_label_coords(1.02, 0.025)
            plt.yscale("log")
            # using plothelper() read the data for the graph
            h_values, dh_values, ddh_values = plothelper(filename_1)
            square = [(float(x)+0.0001)**2 for x in h_values]
            # draw plots for dh_values and ddh_values
            plt.plot(h_values, dh_values, "b", linewidth = 2, linestyle="solid")
            plt.plot(h_values, ddh_values, "g", linewidth = 2, linestyle="solid")
            # draw guideline
            plt.plot(h_values, square, "grey", linewidth = 2,
                     linestyle="dashdot")
            plt.plot(h_values, [float(x)-10**-10 for x in h_values], "grey", linewidth = 2,
                     linestyle="dotted")
            # append graphs to legend
            legend.append("dh")
            graphs.append(Line2D([0], [0], color = "b", linewidth=2))
            legend.append("ddh")
            graphs.append(Line2D([0], [0], color = "g", linewidth=2))
            # append guideline to legend
            legend.append("h")
            graphs.append(Line2D([0], [0], color = "grey", linewidth=2, linestyle="dotted"))
            legend.append("h^2")
            graphs.append(Line2D([0], [0], color = "grey", linewidth=2, linestyle="dashdot"))
            # rotate ticklabels to prevent possible overlapping
            plt.setp(ax1.get_xticklabels(), rotation=20, horizontalalignment='right')
        else:
            ax1.set_ylim(bottom = 10**-13)
            # change graph accordingly
            plt.ylabel("y", fontsize = 20, rotation = 0)
            ax1.yaxis.set_label_coords(-0.01, 1)
            plt.xlabel("h", fontsize = 20)
            ax1.xaxis.set_label_coords(1.02, 0.025)
            plt.yscale("log")
            # using plothelper() read the data for the graph
            h_values, dhs_values, _ = plothelper(filename_1)
            quartic = [(float(x)+0.0001)**4 for x in h_values]
            # draw plots for dh_values and ddh_values
            plt.plot(h_values, dhs_values, "b", linewidth = 2, linestyle="solid")
            # draw guideline
            plt.plot(h_values, quartic, "grey", linewidth = 2, linestyle="dashdot")
            # append graphs to legend
            legend.append("dhs")
            graphs.append(Line2D([0], [0], color = "b", linewidth=2))
            # append guideline to legend
            legend.append("h^4")
            graphs.append(Line2D([0], [0], color = "grey", linewidth=2, linestyle="dashdot"))
            # rotate ticklabels to prevent overlapping
            plt.setp(ax1.get_xticklabels(), rotation=20, horizontalalignment='right')
        # create legend
        plt.legend(graphs, legend, fontsize=20, loc="upper left")
        # show drawn plot
        plt.show()
    except TypeError as exc:
        raise exc



def progress_bar(current : int, total : int, bar_length=20):
    '''support function for main(); displays a progressbar in console based on the percentile
    calculated by current / total
    
    Parameter
    ---------
    current : int
        current number
    total : int
        maximum number to be reached
    bar_length : int
        constant for resizing the progress bar
    
    Exceptions

    ----------
    ValueError
        current is bigger than total
        current is negative
    TypeError
        current is not an int
        total is not an int
        bar_length is not an int
    '''
    try:
        # test if input is correct
        if current < 0 or current > total:
            raise ValueError
        # compute percentage
        fraction = current / total
        # save according arrow length and fill the rest with " "
        arrow = int(fraction * bar_length - 1) * '-' + '>'
        padding = int(bar_length - len(arrow)) * ' '
        # test if the progressbar is full and enable or deny overwrighting of the bar accordingly
        if current == total:
            ending = '\n'
        else:
            ending = '\r'
        # print the progressbar
        print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)
    except TypeError as exc:
        raise exc
    except ValueError as exc:
        print(f"\n{exc}: Progressbar is overfilled! current value: {current}")




def main():
    """ Example of code that can be run using the provided class, mehtods and function
    """
    # initialize and declare variables
    a = math.pi         # pylint: disable=invalid-name
    b = 3*math.pi       # pylint: disable=invalid-name
    p = 1000            # pylint: disable=invalid-name
    k_value = 0.1       # pylint: disable=invalid-name

    # update / create progressbar
    progress = 0
    progress_bar(progress, 63)

    # clear / create files needed for saving data
    for file in ["_1", "_2", "_dh", "_ddh", "_error", "_dhs_error"]:
        ml.clear("g", file + ".csv")
        # update progressbar
        progress+=1
        progress_bar(progress, 63)

    # create function and analytic derivatives for testing code
    def func_g(x_value):
        return math.sin(x_value)/x_value
    def func_d_g(x_value):
        return math.cos(x_value)/x_value - math.sin(x_value)/x_value**2
    def func_dd_g(x_value):
        return -((x_value**2-2)*math.sin(x_value)+2*x_value*math.cos(x_value))/x_value**3

    # update progressbar
    progress+=1
    progress_bar(progress, 63)

    # create function and analytic derivatives for testing code
    def func_gk(x_value, k=k_value):
        return math.sin(k*x_value)/x_value
    def func_d_gk(x_value, k=k_value):
        return (k*math.cos(k*x_value)/x_value) - (math.sin(k*x_value)/x_value**2)
    def func_dd_gk(x_value, k=k_value):
        return -((k**2*x_value**2-2)*math.sin(k*x_value)+2*k*x_value*math.cos(k*x_value))/x_value**3

    # update progressbar
    progress+=1
    progress_bar(progress, 63)

    # plot approximations of derivatives
    plot_derivatives(0.01, a, b, p, func_g)

    # update progressbar
    progress+=1
    progress_bar(progress, 63)

    # plot approximations of derivatives and analytic derivatives
    plot_derivatives(0.01, a, b, p, func_g, func_d_g, func_dd_g)

    # update progressbar
    progress+=1
    progress_bar(progress, 63)

    # plot approximations of derivatives
    plot_derivatives(0.01, a, b, p, func_gk)

    # update progressbar
    progress+=1
    progress_bar(progress, 63)

    # plot approximations of derivatives and analytic derivatives
    plot_derivatives(0.01, a, b, p, func_gk, func_d_gk, func_dd_gk)

    # update progressbar
    progress+=1
    progress_bar(progress, 63)

    # plot f(x)=x, f(x)=x^2, f(x)=x^3 and derivatives of these
    power_func(np.linspace(a, b, p))

    # update progressbar
    progress+=1
    progress_bar(progress, 63)

    for i in [0, 10, 14]:
        h = 10**(-i)        # pylint: disable=invalid-name
        exp_1 = FiniteDifference(h, func_g, func_d_g, func_dd_g)
        exp_1.experiment("g", a, b, p)
        # update progressbar
        progress+=1
        progress_bar(progress, 63)

    # plot different approximations of first derivative and analytic first derivative
    plot("1. Ableitung", "g_dh.csv", "dh", "g_1.csv", "g'")

    # update progressbar
    progress+=1
    progress_bar(progress, 63)

    # since the approximation for the first approximated derivative works better on a different
    # intervall than the second approximation we draw them in their own plot. Therefore we need to
    # save new data. In order to not corrupt this data the files must first be cleared

    # clear files needed for saving data
    for file in ["_1", "_2", "_dh", "_ddh", "_error", "_dhs_error"]:
        ml.clear("g", file + ".csv")
        # update progressbar
        progress+=1
        progress_bar(progress, 63)

    # generate data for plotting second approximated derivative
    for i in [0, 4, 7]:
        h = 10**(-i)        # pylint: disable=invalid-name
        exp_1 = FiniteDifference(h, func_g, func_d_g, func_dd_g)
        exp_1.experiment("g", a, b, p)
        # update progressbar
        progress+=1
        progress_bar(progress, 63)

    # plot different approximations of second derivative and analytic first derivative
    plot("2. Ableitung", "g_ddh.csv", "ddh", "g_2.csv", "g''" )

    # update progressbar
    progress+=1
    progress_bar(progress, 63)

    # Since we want to draw both plots at once we need to show a bigger intervall in our plot.
    # Therefore new data must be saved and we again first need to clear the old data from the
    # files.

    # clear files needed for saving data
    for file in ["_1", "_2", "_dh", "_ddh", "_error", "_dhs_error"]:
        ml.clear("g", file + ".csv")
        # update progressbar
        progress+=1
        progress_bar(progress, 63)

    # generate data for plotting the error of the approximations
    # we used range(0, 15) insted of [0, 4, 8, 12, 15] for our experiments
    for i in [0, 4, 8, 12, 15]:
        h = 10**(-i)        # pylint: disable=invalid-name
        exp_1 = FiniteDifference(h, func_g, func_d_g, func_dd_g)
        exp_1.experiment("g", a, b, p)
        # update progressbar
        progress+=1
        progress_bar(progress, 63)

    # plot error of the approximations relative to h
    plot("Approximationsfehler", "g_error.csv", "")

    # update progressbar
    progress+=1
    progress_bar(progress, 63)

    # Since we want to generate new data we again need to delete old saves.

    # clear files needed for saving data
    for file in ["_1", "_2", "_dh", "_ddh", "_error", "_dhs", "_dhs_error"]:
        ml.clear("g", file + ".csv")
        # update progressbar
        progress+=1
        progress_bar(progress, 63)

    # generate data for plotting first approximated derivative using an alternative function
    for i in [-0.3, 0, 2]:
        h = 10**(-i)        # pylint: disable=invalid-name
        exp_1 = FiniteDifference(h, func_g, func_d_g, func_dd_g)
        exp_1.experiment("g", a, b, p)
        # update progressbar
        progress+=1
        progress_bar(progress, 63)

    # plot alternative approximation of first derivative
    plot("1. Ableitung(Alternative)", "g_dhs.csv", "dhs", "g_1.csv", "g'")

    # update progressbar
    progress+=1
    progress_bar(progress, 63)

    # clear files needed for saving data
    for file in ["_1", "_2", "_dh", "_ddh", "_error", "_dhs", "_dhs_error"]:
        ml.clear("g", file + ".csv")
        # update progressbar
        progress+=1
        progress_bar(progress, 63)

    # generate data for plotting the error of the alternative approximations
    # we used range(0, 15) insted of [0, 4, 8, 12, 15] for our experiments
    for i in [0, 4, 8, 12, 15]:
        h = 10**(-i)        # pylint: disable=invalid-name
        exp_1 = FiniteDifference(h, func_g, func_d_g, func_dd_g)
        exp_1.experiment("g", a, b, p)
        # update progressbar
        progress+=1
        progress_bar(progress, 63)

    # plot error of alternative approximation of first derivative
    plot("Approximationsfehler", "g_dhs_error.csv", "")

    # update progressbar
    progress += 1
    progress_bar(progress, 63)



# main-guard
if __name__=="__main__":
    main()
