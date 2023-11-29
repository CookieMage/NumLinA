''' Gruppe: 21
This is an implementation of the class BlockMatrix, which can be used to approximate the
solution of the Poisson problem.

classes
-------
BlockMatrix
    class for representing matrices used to solve the poisson-problem

functions
---------
main()
    Example of code that can be run using the provided functions
'''
from scipy import sparse

class BlockMatrix:
    '''class for representing matrices used to solve the poisson-problem

    Parameters
    ----------
    d : int
        dimension of the space
    n : int
        number of intervals in each dimension
    
    Attributes
    ----------
    d : int
        dimension of the space
    n : int
        number of intervals in each dimension
    a_d : sparse.csr
        the constructed matrix as a sparse._csr.csr_matrix

    methods
    -------
    __init__()
        returns an Object of class FiniteDifference
    get_sparse()
        returns the matrix represented by the object as a np.ndarray
    eval_sparsity()
        returns the absolute and relative number of non-zero-entries in the matrix
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
        if not isinstance(n, int):
            raise TypeError('n must be an int')
        if d < 1 or d > 3 or n < 2:
            raise ValueError
        
        self.d = d  # pylint: disable=invalid-name
        self.n = n  # pylint: disable=invalid-name
        
        a_1 = sparse.diags([-1, 2*self.d, -1], [-1, 0, 1], shape=(self.n-1, self.n-1))
        
        # assign the right value to self.a_d
        if d == 1:
            self.a_d = a_1
        else:
            # generate block-matrix, which contains a_1 on the main diagonal
            a_2_block = sparse.block_diag([a_1 for _ in range(self.n-1)])
            # generate block-matrix, which contains the identity on the corresponding diagonal for a_2
            a_2_ident = sparse.diags([-1, -1], [-(self.n-1), self.n-1],
                                     shape=((self.n-1)**2, (self.n-1)**2))
            # add matrices to get a_2
            a_2 = a_2_block + a_2_ident
            
            if d == 2:
                self.a_d = a_2
            else:
                # generate block-matrix, which contains a_2 on the main diagonal
                a_3_block = sparse.block_diag([a_2 for _ in range(self.n-1)])
                # generate block-matrix, which contains the identity on the corresponding diagonal for a_3
                a_3_ident = sparse.diags([-1, -1], [-(self.n-1)**2, (self.n-1)**2],
                                         shape=((self.n-1)**3, (self.n-1)**3))
                # add matrices to get a_3
                self.a_d = a_3_block + a_3_ident

    def get_sparse(self):
        '''method for returning the matrix as an array

        Returns
        -------
        np.ndarray
            represented matrix
        '''
        return self.a_d

    def eval_zeros(self):
        '''method for getting the absolute and relative number of non-zeros-entries

        Returns
        -------
        int
            absolute number of non-zero-entries
        float
            relative number of non-zero-entries
        '''
        abs_non_zero = self.a_d.count_nonzero()
        abs_entries = ((self.n-1)**self.d)**2
        rel_non_zero = abs_non_zero / abs_entries
        return abs_non_zero, rel_non_zero

def main():
    '''Example of code that can be run using the provided class and methods
    '''
    mat_1 = BlockMatrix(2, 4)

    print(mat_1.get_sparse().toarray())
    print(mat_1.eval_zeros())

    mat_2 = BlockMatrix(3, 5)

    print(mat_2.get_sparse().toarray())
    print(mat_2.eval_zeros())

if __name__ == "__main__":
    main()

# Der Datentyp von a_d ist scipy.sparse._csr.csr_array und der Test verlangt etwas anderes.
# wir konnten das geforderte format nicht erzeugen aber meinen, dass dieses gleichbedeutend sein
# sollte.