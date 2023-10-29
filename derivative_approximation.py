import math
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import manage_list as ml



class FiniteDifference:
    """ Represents the first and second order finite difference approximation of a function and
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
    """
    #weitere attribute in den Docstring
    h = None
    f = None
    d_f = None
    dd_f = None

    def __init__(self, h, f, d_f=None, dd_f=None):
        """ returns an Object of class FiniteDifference

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
        ValueError
            One of the given arguments is of the wrong type
        """
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
        self.h = h
        self.f = f
        self.d_f = d_f
        self.dd_f = dd_f


    def compute_dh_f(self):
        """Calculates the approximation for the first derivative of the f with step size h.

        Parameters
        ----------
        -

        Return
        ------
        callable
            Calculates the approximation of the first derivative for a given x.
        """
        def f_1(x : float):
            return (self.f(x+self.h)-self.f(x))/self.h
        return f_1

    def compute_ddh_f(self):
        """Calculates the approximation for the second derivative of f with step size h.
        Parameters
        ----------
        -
        Return
        ------
        callable
            Calculates the approximation of the first derivative for a given x.
        """
        def f_2(x : float):
            return (self.f(x+self.h)-2*self.f(x)+self.f(x-self.h))/(self.h**2)
        return f_2

    def compute_errors(self, a, b, p):
        """ Calculates an approximation to the errors between an approximation
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
            If no analytic derivative was provided by the user.
        """
        if self.d_f is None and self.dd_f is None:
            raise ValueError
        f_1 = self.compute_dh_f()
        f_2 = self.compute_ddh_f()
        max_dif_1 = 0.0
        max_dif_2 = 0.0
        steps = np.linspace(a, b, p)
        if self.d_f is not None:
            for i in steps:
                max_dif_1 = max(max_dif_1, abs(f_1(i) - self.d_f(i)))
        if self.d_f is not None:
            for i in steps:
                max_dif_2 = max(max_dif_2, abs(f_2(i) - self.dd_f(i)))
        return max_dif_1, max_dif_2


    def experiment(self, func_name, a, b, p):
        steps = np.linspace(a, b, p)
        dh_f = self.compute_dh_f()
        ddh_f = self.compute_ddh_f()
        errors = self.compute_errors(a, b, p)
        ml.save(func_name, "_error.csv", "dh_1", self.h, errors[0])
        ml.save(func_name, "_error.csv", "ddh_2", self.h, errors[1])
        ml.save(func_name, "_error.csv", "")
        for i in steps:
            ml.save(func_name, "_dh.csv", i, dh_f(i))
            ml.save(func_name, "_ddh.csv", i, ddh_f(i))
            ml.save(func_name, "_1.csv", i, self.d_f(i))
            ml.save(func_name, "_2.csv", i, self.dd_f(i))
        ml.save(func_name, ".csv", "")
        ml.save(func_name, "_dh.csv", "")
        ml.save(func_name, "_ddh.csv", "")
        ml.save(func_name, "_1.csv", "")
        ml.save(func_name, "_2.csv", "")




def plothelper(filename,):
    with open("experiments/" + filename, "r", encoding = "utf8") as file:
        lines = file.readlines()
        lines = [i.split() for i in lines]
        lines_list = []

        if "error" not in filename:
            counter = 0
            for i,e in enumerate(lines):
                if len(lines_list) < counter+1:
                    lines_list += [[]]
                if e == []:
                    counter += 1
                    continue
                lines_list[counter] += e
            x, y = [], []
            for e in lines_list:
                x += [[e[2*j] for j in range(len(e)//2)]]
            for e in lines_list:
                y += [[e[2*j+1] for j in range(len(e)//2)]]
            for i,e in enumerate(y):
                for j,f in enumerate(e):
                    y[i][j] = float(f)
            x = x[0]
            for i,e in enumerate(x):
                x[i] = float(e)
            return x, y
        else:
            counter = 0
            for i,e in enumerate(lines):
                if len(lines_list) < counter+1:
                    lines_list += [[]]
                if e == []:
                    counter += 1
                    continue
                lines_list[counter] += e
            for i,e in enumerate(lines_list):
                for j,f in enumerate(e):
                    if "dh" in f:
                        lines_list[i].pop(j)
            h, temp, dh, ddh = [], [], [], []
            for e in lines_list:
                h += [[e[2*j] for j in range(len(e)//2)]]
            h = [e[0] for e in h]
            for e in lines_list:
                temp += [[e[2*j+1] for j in range(len(e)//2)]]
            for e in temp:
                dh += [[e[2*j] for j in range(len(e)//2)]]
            for e in temp:
                ddh += [[e[2*j+1] for j in range(len(e)//2)]]
            dh = [inner for outer in dh for inner in outer]
            ddh = [inner for outer in ddh for inner in outer]
            for i, e in enumerate(dh):
                dh[i] = float(e)
            for i, e in enumerate(ddh):
                ddh[i] = float(e)
            return h, dh, ddh



def plot(title, filename_1, name_1, filename_2 = None, name_2 = None, filename_3 = None, name_3 = None, filename_4 = None, name_4 = None):
    _, ax1 = plt.subplots(figsize=(5, 5))
    plt.xticks(fontsize=17)
    plt.yticks(fontsize=17)
    plt.xlabel("x", fontsize = 20)
    ax1.xaxis.set_label_coords(1.02, 0.025)
    ax1.yaxis.get_offset_text().set_fontsize(20)
    plt.title(title, fontsize=40)
    ax1.grid()
    legend = []
    graphs = []

    y1 = []
    if "error" not in filename_1:
        x, y1 = plothelper(filename_1)
        for e in y1:
            print(max(e))
            plt.plot(x, e, "b", linewidth = 2, linestyle="dashed")
        legend.append(name_1)
        graphs.append(Line2D([0], [0], color = "b", linewidth=2, linestyle="dashed"))

        if filename_2 is not None:
            y2 = []
            x, y2 = plothelper(filename_2)
            for e in y2:
                plt.plot(x, e, "g", linewidth = 2, linestyle="solid")
            legend.append(name_2)
            graphs.append(Line2D([0], [0], color = "g", linewidth=2, linestyle="solid"))

        if filename_3 is not None:
            y3 = []
            x, y3 = plothelper(filename_3)
            for e in y3:
                plt.plot(x, e, "r", linewidth = 2, linestyle="dotted")
            legend.append(name_3)
            graphs.append(Line2D([0], [0], color = "r", linewidth=2, linestyle="dotted"))

        if filename_4 is not None:
            y4 = []
            x, y4 = plothelper(filename_4)
            for e in y4:
                plt.plot(x, e, "y", linewidth = 2, linestyle="dashdot")
            legend.append(name_4)
            graphs.append(Line2D([0], [0], color = "y", linewidth=2, linestyle="dashdot"))
    else:
        plt.xlabel("h", fontsize = 20)
        ax1.xaxis.set_label_coords(1.02, 0.025)
        plt.yscale("log")
        plt.xscale("log")
        h, dh, ddh = plothelper(filename_1)
        plt.plot(h, dh, "b", label = "dh", linewidth = 2, linestyle="solid")
        plt.plot(h, ddh, "g", label = "ddh", linewidth = 2, linestyle="solid")
        legend.append("dh")
        graphs.append(Line2D([0], [0], color = "b", linewidth=2))
        legend.append("ddh")
        graphs.append(Line2D([0], [0], color = "g", linewidth=2))
    
    plt.setp(ax1.get_xticklabels(), rotation=40, horizontalalignment='right')
    plt.legend(graphs, legend, fontsize=20, loc="upper left")
    plt.show()

def power_func(numbers : list):
    colors = ["b", "g", "y"]
    line = ["solid", "dashed", "dotted"]
    for s in [1, 2, 3]:
        #labels = [f"x^{s}", f"{s}x^{s-1}", f"{s-1*s}x^{s-2}"]
        labels = [f"f_{s}", f"f_{s}'", f"f_{s}''"]
        def f(x):
            return x**s
        def f_1(x):
            return s*x**(s-1)
        def f_2(x):
            return (s-1)*s*x**(s-2)
        for i,e in enumerate([f, f_1, f_2]):
            plt.plot(numbers, [e(x) for x in numbers], label = labels[i],
                     color = colors[s-1], linewidth = 2, linestyle = line[i])
            plt.legend(fontsize = 20)


def progress_bar(current, total, bar_length=20):
    '''support function for main(); displays a progressbar in console based on the percentile
    calculated by current / total
    
    Parameter
    ---------
    current
        current number
    total
        maximum number to be reached
    bar_length
        constant for resizing the progress bar
    
    Exceptions
    ----------
    ValueError
        current is bigger than total
    TypeError
        current is not an int
        total is not an int
        bar_length is not an int
    '''
    try:
        # teste ob die Eingabe korrekt ist
        if current > total:
            raise ValueError
        # berechne den prozentualen Fortschritt
        fraction = current / total
        # speichere die dazugehoerige Pfeillaenge und fuelle den rest mit Leerzeichen auf
        arrow = int(fraction * bar_length - 1) * '-' + '>'
        padding = int(bar_length - len(arrow)) * ' '
        # teste ob das Ende der Progressbar erreicht wurde und ermoegliche eine ueberschreibung bzw
        # verhindere diese
        if current == total:
            ending = '\n'
        else:
            ending = '\r'
        # gebe den Fortschritt in Form der Progressbar aus
        print(f'Progress: [{arrow}{padding}] {int(fraction*100)}%', end=ending)
    except TypeError as exc:
        raise exc



def main():
    """ Example of code that can be run using the provided class, mehtods and function
    """
    a = math.pi
    b = 3*math.pi
    p = 10
    c = 0.1

    for func_name in ["g", "gk"]:
        for file in ["_1", "_2", "_dh", "_ddh", "_error", ""]:
            ml.clear(func_name, file + ".csv")


    def g(x):
        return math.sin(x)/x
    def d_g(x):
        return math.cos(x)/x - math.sin(x)/x**2
    def dd_g(x):
        return -((x**2-2)*math.sin(x)+2*x*math.cos(x))/x**3

    def gk(x, k=c):
        return math.sin(k*x)/x
    def d_gk(x, k=c):
        return (k*math.cos(k*x)/x) - (math.sin(k*x)/x**2)
    def dd_gk(x, k=c):
        return -((k**2*x**2-2)*math.sin(k*x)+2*k*x*math.cos(k*x))/x**3



    for i in range(1, 16):
        h = 10**(-i)
        exp_1 = FiniteDifference(h, g, d_g, dd_g)
        exp_1.experiment("g", a, b, p)
    for i in range(1, 16):
        h = 10**(-i)
        exp_2 = FiniteDifference(h, gk, d_gk, dd_gk)
        exp_2.experiment("gk", a, b, p)


    plot("1. Ableitung", "g_dh.csv", "dh", "g_1.csv", "g'")
    plot("2. Ableitung", "g_ddh.csv", "ddh", "g_2.csv", "g''" )
    plot("Abschätzungsfehler", "g_error.csv", "")
    plot("1. Ableitung", "gk_dh.csv", "dh", "gk_1.csv", "g'")
    plot("2. Ableitung", "gk_ddh.csv", "ddh", "gk_2.csv", "g''")
    plot("Abschätzungsfehler", "gk_error.csv", "")


if __name__=="__main__":
    main()
