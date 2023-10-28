import manage_list as ml
import matplotlib.pyplot as plt
import re
import os


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
    #wirklich richtig so???


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
        def f_1(y : float):
            return (self.f(y+self.h)-self.f(y))/self.h
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
        def f_2(x):
            return (self.f(x+self.h)-2*self.f(x)+self.f(x-self.h))/self.h**2
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
        # Bei der Brechnung von step muss p-1 genutz werden, damit später b+1 in range verwendet
        # werden kann und die korrekte Anzahl an Intervallen vorhanden bleibt
        step = ((2*abs(b-a)) // (p-1))
        if step == 0:
            step = 1
        # Es wird 2*abs(b-a)+1 als intervallende genutzt, damit wir durch 0.5*x eine höhere
        # Genauigkeit der Intervallaufteilung genutzt werden kann und range() kein abgeschlossenes
        # Intervall erzeugt. Damit b also im Intervall sein kann müssen wir das Intervallende um 1
        # erhöhen. Dies funktioniert, da range() nur ganze Zahlen erzeugen kann.
        array = [a+0.5*x for x in range(0, 2*abs(b-a)+1, step)]
        if self.d_f is not None:
            for i in array:
                max_dif_1 = max(max_dif_1, abs(f_1(i) - self.d_f(i)))
        if self.d_f is not None:
            for i in array:
                max_dif_2 = max(max_dif_2, abs(f_2(i) - self.dd_f(i)))
        return max_dif_1, max_dif_2


    def experiment(self, func_name, a, b, p):
        step = ((2*abs(b-a)) // (p-1))
        if step == 0:
            step = 1
        array = [a+0.5*x for x in range(0, 2*abs((b+1)-a), step)]
        dh_f = self.compute_dh_f()
        ddh_f = self.compute_ddh_f()
        errors = self.compute_errors(a, b, p)
        ml.save(func_name, "errors.csv", "dh_1", errors[0])
        ml.save(func_name, "errors.csv", "ddh_2", errors[1])
        ml.save(func_name, "errors.csv", "")
        for i in array:
            ml.save(func_name, ".csv", i, self.f(i))
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
    with open("experiments/" + filename, "r") as file:
        lines = file.readlines()
        lines = [i.split() for i in lines]
        lines_list = []
        
        counter = 0
        for i,e in enumerate(lines):
            if len(lines_list) < counter+1:
                lines_list += [[]]
            if e == []:
                counter += 1
                continue
            lines_list[counter] += e
        print(filename)
        print(lines_list[0])
        x, y = [], []
        for i,e in enumerate(lines_list):
            x += [[e[2*i] for i in range(len(e)//2)]]
        for i in enumerate(lines_list):
            y += [[e[2*i+1] for i in range(len(e)//2)]]
        for i,e in enumerate(y):
            print(y[0])
            y[i] = float(e)
        for i,e in enumerate(x):
            print(x)
            x[i] = float(e)
    return x, y



def plot(title, filename_1, name_1, filename_2 = None, name_2 = None, filename_3 = None, name_3 = None, filename_4 = None, name_4 = None):
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize=40)
    plt.xlabel("Anzahl der Listenelemente", fontsize = 20)
    ax1.grid()
    
    y1 = []
    x, y1 = plothelper(filename_1)
    plt.plot(x, y1, "b", label = name_1, linewidth = 2, linestyle="solid")
    
    if filename_2 != None:
        y2 = []
        x, y2 = plothelper(filename_2)
        plt.plot(x, y2, "g", label = name_2, linewidth = 2, linestyle="dashed")
        
    if filename_3 != None:
        y3 = []
        x, y3 = plothelper(filename_3)
        plt.plot(x, y3, "r", label = name_3, linewidth = 2, linestyle="dotted")
        
    if filename_4 != None:
        y4 = []
        x, y4 = plothelper(filename_4)
        plt.plot(x, y4, "y", label = name_4, linewidth = 2, linestyle="dashdot")
        
    plt.legend(fontsize=20)
    plt.show()


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


# Methoden / Funktionen 3 un 4

def main():
    """ Example of code that can be run using the provided class, mehtods and function
    """
    
    a = -1000
    b = 1000
    p = 10

    for i in ["f1", "f2", "f3"]:
        ml.clear(i, "errors.csv")
        ml.clear(i, ".csv")
        ml.clear(i, "_dh.csv")
        ml.clear(i, "_ddh.csv")
        ml.clear(i, "_1.csv")
        ml.clear(i, "_2.csv")


    def f_1(x):
        return x**2
    def d_f_1(x):
        return 2*x
    def dd_f_1(x):
        return 2

    def f_2(x):
        return 3*x**3
    def d_f_2(x):
        return 9*x**2
    def dd_f_2(x):
        return 18*x

    def f_3(x):
        return 3*x**2+2*x-7
    def d_f_3(x):
        return 6*x+2
    def dd_f_3(x):
        return 6
    
    for i in range(10):
        h = 10**(-i)
        exp_1 = FiniteDifference(h, f_1, d_f_1, dd_f_1)
        exp_1.experiment("f1", a, b, p)
    for i in range(10):
        h = 10**(-i)
        exp_2 = FiniteDifference(h, f_2, d_f_2, dd_f_2)
        exp_2.experiment("f2", a, b, p)
    for i in range(10):
        h = 10**(-i)
        exp_3 = FiniteDifference(h, f_3, d_f_3, dd_f_3)
        exp_3.experiment("f3", a, b, p)

    plot("1. Ableitung", "f1_1.csv", "f'", "f1_dh.csv", "dh")
    plot("2. Ableitung", "f1_1.csv", "f''", "f1_dh.csv", "ddh")
    plot("1. Ableitung", "f2_1.csv", "f'", "f2_dh.csv", "dh")
    plot("2. Ableitung", "f2_1.csv", "f''", "f2_dh.csv", "ddh")
    plot("1. Ableitung", "f3_1.csv", "f'", "f3_dh.csv", "dh")
    plot("2. Ableitung", "f3_1.csv", "f''", "f3_dh.csv", "ddh")


if __name__=="__main__":
    main()
