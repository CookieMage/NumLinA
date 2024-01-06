''' Gruppe: 21

This is an implementation of the class BlockMatrix, which can be used to approximate the
solution of the Poisson problem.

classes
-------
BlockMatrix
    class for representing matrices used to solve the Poisson problem

functions
---------
graph_sparse_dense()
    creates a plot representing the number of non-zero-entries for sparse and
    non-sparse matrices of type BlockMatrix
graph_lu()
    creates a plot representing the number of non-zero-entries for a sparse matrix
    of type BlockMatrix and a lu-matrix
main()
    Example of code that can be run using the provided functions
'''
from scipy import sparse
from scipy import linalg
import numpy as np
from plotter import plotter

class BlockMatrix:
    '''Represents block matrices arising from finite difference approximations
    of the Laplace operator.

    Parameters
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.

    Attributes
    ----------
    d : int
        Dimension of the space.
    n : int
        Number of intervals in each dimension.
    a_d : sparse._csr.csr_matrix
        Matrix created based on d and n

    methods
    -------
    __init__()
        returns an Object of class FiniteDifference
    get_sparse()
        returns the matrix represented by the object as a np.ndarray
    eval_sparsity()
        returns the absolute and relative number of non-zero-entries in the matrix
    get_lu()
        returns the lu-decomposition of a_d as np.ndarrays
    eval_sparsity_lu()
        returns the absolute and relative number of non-zero-entries in the matrix l+u based on
        some restrictions

    Raises
    ------
    ValueError
        If d < 1 or n < 2.
    '''
    d = None
    n = None
    a_d = None

    def __init__(self, d : int, n : int):   # pylint: disable=invalid-name
        '''returns an Object of class FiniteDifference

        Parameters
        ----------
        d : int
            dimension of the space
        n : int
            number of intervals in each dimension

        Raises
        ------
        TypeError
            d or n was not an int
        ValueError
            d was < 1 or > 3 or n was < 2
        '''
        if not isinstance(d, int):
            raise TypeError('d must be an int')
        if not isinstance(n, (int, np.integer)):
            raise TypeError('n must be an int')
        if d < 1 or d > 3 or n < 2:
            raise ValueError

        self.d = d  # pylint: disable=invalid-name
        self.n = int(n)  # pylint: disable=invalid-name

        a_1 = sparse.diags([-1, 2*self.d, -1], [-1, 0, 1], shape=(self.n-1, self.n-1))

        # assign the right value to self.a_d
        if d == 1:
            self.a_d = a_1
        else:
            # generate block-matrix, which contains a_1 on the main diagonal
            a_2_block = sparse.block_diag([a_1 for _ in range(self.n-1)])
            # generate block-matrix, which contains the identity on the corresponding diagonal for
            # a_2
            a_2_ident = sparse.diags([-1, -1], [-(self.n-1), self.n-1],
                                     shape=((self.n-1)**2, (self.n-1)**2))
            # add matrices to get a_2
            a_2 = a_2_block + a_2_ident

            if d == 2:
                self.a_d = a_2
            else:
                # generate block-matrix, which contains a_2 on the main diagonal
                a_3_block = sparse.block_diag([a_2 for _ in range(self.n-1)])
                # generate block-matrix, which contains the identity on the corresponding
                # diagonal for a_3
                a_3_ident = sparse.diags([-1, -1], [-(self.n-1)**2, (self.n-1)**2],
                                         shape=((self.n-1)**3, (self.n-1)**3))
                # add matrices to get a_3
                self.a_d = a_3_block + a_3_ident


    def get_sparse(self):
        """ Returns the block matrix as sparse matrix.

        Returns
        -------
        scipy.sparse.csr_matrix
            The block_matrix in a sparse data format.
        """
        return sparse.csr_matrix(self.a_d)


    def eval_sparsity(self):
        """ Returns the absolute and relative numbers of non-zero elements of the matrix. The
        relative quantities are with respect to the total number of elements of the represented
        matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """
        # calculate non-zero-entries of the matrix
        abs_non_zero = (self.n-1)**self.d+2*self.d*(self.n-2)*(self.n-1)**(self.d-1)
        # calculate number of entries of the matrix
        abs_entries = ((self.n-1)**self.d)**2
        rel_non_zero = abs_non_zero / abs_entries
        return abs_non_zero, rel_non_zero


    def get_lu(self):
        """ Provides an LU-Decomposition of the represented matrix A of the form A = p * l * u

        Returns
        -------
        p : numpy.ndarray
            permutation matrix of LU-decomposition
        l : numpy.ndarray
            lower triangular unit diagonal matrix of LU-decomposition
        u : numpy.ndarray
            upper triangular matrix of LU-decomposition
        """
        return linalg.lu(self.a_d.toarray(), permute_l=False)


    def eval_sparsity_lu(self):
        """ Returns the absolute and relative numbers of non-zero elements of the LU-Decomposition.
        The relative quantities are with respect to the total number of elements of the represented
        matrix.

        Returns
        -------
        int
            Number of non-zeros
        float
            Relative number of non-zeros
        """
        lu = self.get_lu()  #pylint: disable=invalid-name
        # create negativ identity
        sub = sparse.diags([-1], [0], shape = ((self.n-1)**self.d, (self.n-1)**self.d))
        # create matrix as specified above
        result = lu[0] + lu[1] + sub
        # get number non-zero-entries and entries
        abs_non_zero = np.count_nonzero(result)
        abs_entries = ((self.n-1)**self.d)**2
        # calculate relative number of non-zero-entries
        rel_non_zero = abs_non_zero / abs_entries
        return abs_non_zero, rel_non_zero


def graph_sparse_dense(maximum=2, n=25, dim = [1,2,3]):   #pylint: disable=invalid-name, disable=dangerous-default-value
    '''creates a plot representing the number of non-zero-entries for sparse and non-sparse
    matrices of type BlockMatrix; The data is calculated to reduce runtime and needed resources

    Parameters
    ----------
    maximum : int, optional
        10**maximum is the maximum number slices for a BlockMatrix, by default 2
    n : int, optional
        describes the number of datapoints for each graph, by default 25
    dim : list, optional
        list of dimensions that should be shown in the plot, by default [1,2,3]
    '''
    # create logarithmic list of int from 10^0.4 to 10^maximum
    x_values = np.logspace(0.4, maximum, dtype=int, num=n)
    # convert numpy.int to int in order to prevent stackoverflow and format a list in order to
    # setup later computations
    x_values = [[int(x)**3 for x in x_values], [int(x)**1.5 for x in x_values],
                [int(x) for x in x_values]]

    # create lists for saving data for the plot
    data = []
    labels = []

    for i,d in enumerate(dim):  #pylint: disable=invalid-name
        data += [[],[]]
        labels += [f"sparse d={d}", f"dense d={d}"]
        # save sparsity information for each value of n
        for n in x_values[d-1]: #pylint: disable=redefined-argument-from-local
            abs_non_zero = (n-1)**d+2*d*(n-2)*(n-1)**(d-1)

            # save sparsity information
            data[2*i] += [abs_non_zero]
            data[2*i+1] += [((n-1)**d)**2]

    # create lists for plotting the data
    linestyles = ["dashdot", "dotted"]*3
    # if only on dimension is plotted make both of the plots a different color
    if len(dim) == 1:
        colors = ["b", "r"]
    else:
        # make every graph of one dimension the same color
        colors = ["b", "b", "r", "r", "c", "c"]

    # plot data
    x_values = [x_values[0], x_values[0], x_values[1], x_values[1], x_values[2], x_values[2]]
    plotter(x_values, data, labels, linestyles, colors)


def graph_lu(maximum=1, n=10):  #pylint: disable=invalid-name
    '''creates a plot representing the number of non-zero-entries for a sparse matrix of type
    BlockMatrix and a lu-matrix. The lu-matrix is saved in a single matrix by adding them and
    subtracting the identity-matrix.

    Parameters
    ----------
    maximum : int, optional
        10**maximum is the maximum number slices for a BlockMatrix, by default 2
    n : int, optional
        describes the number of datapoints for each graph, by default 25
    '''
    # create logarithmic list of int from 10^0.4 to 10^maximum
    x_values = np.logspace(0.4, maximum, dtype=int, num=n)
    # convert numpy.int to int in order to prevent stackoverflow and format a list in order to
    # setup later computations
    x_values = [[int(x)**3 for x in x_values], [int(x**1.5) for x in x_values],
                [int(x) for x in x_values]]

    # create lists for saving data for the plot
    data_lu = [[],[],[]]
    data_sparse = [[], [], []]

    for d in range(1, 4):   #pylint: disable=invalid-name
        for n in x_values[d-1]: #pylint: disable=redefined-argument-from-local
            # create matrix for evaluating the sparsity of the lu-decomposition
            mat = BlockMatrix(d, n)
            absolute, _ = mat.eval_sparsity_lu()
            # calculate number of non-zero-entries in the matrix itself
            abs_non_zero = (n-1)**d+2*d*(n-2)*(n-1)**(d-1)
            # save generated data
            data_lu[d-1] += [absolute]
            data_sparse[d-1] += [abs_non_zero]

    # create lists for ploting
    data = data_lu + data_sparse
    labels = ["lu d=1", "lu d=2", "lu d=3", "sparse d=1", "sparse d=2", "sparse d=3"]
    linestyles = ["dotted"]*3 + ["dashdot"]*3
    colors = ["b", "r", "c"]*2

    # plot data
    x_values = [x_values[0], x_values[1], x_values[2], x_values[0], x_values[1], x_values[2]]
    plotter(x_values, data, labels, linestyles, colors)


def main():
    '''Example of code that can be run using the provided class and methods
    '''
    d, n = None, None   #pylint: disable=invalid-name
    print("\n-------------------------MAIN-START-------------------------\n")
    print("Es wird empfohlen d<3 und n<4 zu waehlen, um das Lesen des Terminals zu vereinfachen.")
    while not isinstance(d, int):
        input_text_d = "Bitte geben Sie eine Dimension fuer die zu erstellende Matrix an,"
        input_text_d += " wobei gilt 0<d<4.\n"
        d = input(input_text_d) #pylint: disable=invalid-name
        try:
            d=int(d)    #pylint: disable=invalid-name
            if 0>=d or d>=4:
                d = "Mach ich nicht"    #pylint: disable=invalid-name
        except ValueError:
            continue
    while not isinstance(n, int):
        input_text_n = "Bitte geben Sie die Anzahl an Approximationsintervallen fuer die zu "
        input_text_n += "erstellende Matrix an, wobei gilt 1<n.\n"
        n = input(input_text_n) #pylint: disable=invalid-name
        try:
            n = int(n)  #pylint: disable=invalid-name
            if 0>=n:
                n = "Mach ich nicht"    #pylint: disable=invalid-name
        except ValueError:
            continue
    dimensions = [1,2,3]
    mat_1 = BlockMatrix(d, n)
    sparsity_mat_1 = mat_1.eval_sparsity()
    plu = mat_1.get_lu()
    sparsity_plu = mat_1.eval_sparsity_lu()
    print("Die Matrix sieht folgendermaßen aus:\n", mat_1.get_sparse().toarray())
    print(f"Sie hat {sparsity_mat_1[0]} Nicht-Null-Eintraege. Das entspricht",
          f"{sparsity_mat_1[1]*100}% aller Eintraege.")
    input("Als naechstes wird die LU-Zerlegung ausgegeben. Bitte bestaetigen Sie mit ENTER.")
    print("Die LU-Zerlegung der Matrix sieht folgendermaßen aus:",
          f"\n P=\n{plu[0]}\nL=\n{plu[1]}\nU=\n{plu[2]}")
    print("Wenn P, L und U in einer Matrix gespeichert werden, so hat diese ",
          f"{sparsity_plu[0]} Nicht-Null-Eintraege. Das sind {sparsity_plu[1]*100}% der Matrix.")
    input_text_1 = "Es wird nun eine Grafik zur Darstellung der Nicht-Null-Eintraege bezueglich "
    input_text_1 += "verschiedener Dimensionen und Intervall-Anzahlen erstellt. "
    input_text_1 += "Bitte bestaetigen Sie dies mit ENTER."
    input(input_text_1)
    graph_sparse_dense(dim = dimensions)
    input_text_2 = "Das gleiche geschieht nun zur Anzahl der Nicht-Null-Eintraege der "
    input_text_2 += "LU-Zerlegung. Bitte bestaetigen Sie mit ENTER."
    input(input_text_2)
    graph_lu()
    print("\n--------------------------MAIN-END--------------------------\n")


if __name__ == "__main__":
    main()
