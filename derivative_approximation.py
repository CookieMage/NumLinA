import manage_list as ml
import matplotlib.pyplot as plt


class FiniteDifference(h, f, d_f=None, dd_f=None):
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
        if not isinstance(h, float):
            raise TypeError
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
        def f_1(x):
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
        f_1 = self.compute_dh_f()
        f_2 = self.compute_ddh_f()
        max_dif_1 = 0.0
        max_dif_2 = 0.0
        step = (b-a)/p
        if self.d_f is None and self.dd_f is None:
            raise ValueError
        if self.d_f is not None:
            for i in range(a, b, step):
                max_dif_1 = max(max_dif_1, abs(f_1(i) - self.d_f(i)))
        if self.d_f is not None:
            for i in range(a, b, step):
                max_dif_2 = max(max_dif_2, abs(f_2(i) - self.dd_f(i)))
        return max_dif_1, max_dif_2

def plot(title, txtfile, label1):
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    plt.yscale("log")
    plt.xscale("log")
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title, fontsize=40)
    plt.ylabel("y", fontsize = 20)
    plt.xlabel("x", fontsize = 20)
    ax1.grid()


    lines = []
    with open(txtfile, "r") as f:
        lines = f.readlines()
    lines_list = [i.split() for i in lines]
    y = []
    x = []
    for e in lines_list:
        x += float(e[0])
        y += float(e[1])
    
    ax1.plot(x, y, "g", label=label1, linewidth = 2)
    ax1.legend(fontsize = 20, loc='upper left')

    plt.show()




# Methoden / Funktionen 3 un 4

def main():
    """ Example of code that can be run using the provided class, mehtods and function
    """
    pass

if __name__=="__main__":
    main()
