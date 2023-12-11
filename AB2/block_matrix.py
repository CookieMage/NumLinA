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
from scipy import linalg
import numpy as np

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
        return sparse.csr_array(self.a_d)

    def eval_sparsity(self):
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

    def sparse_swapper(self, mat, a, b, mode="row"):
        """
        Reorders the rows and/or columns in a scipy sparse matrix to the specified order.
        """
        if mode!="row" and mode!="col":
            raise ValueError("mode must be 'row' or 'col'!")
        if max(a,b) > mat.shape[0]:
            raise ValueError("a and b must relate to rows/columns in the matrix!")

        new_order = [x for x in range(a)] + [b] + [x for x in range(a+1, b)] + [a]
        new_order += [x for x in range(b+1, mat.shape[0])]

        new_mat = mat
        if mode == "row":
            ident = sparse.eye(mat.shape[0]).tocoo()
            ident.row = ident.row[new_order]
            new_mat = ident.dot(new_mat)
        if mode == "col":
            ident = sparse.eye(mat.shape[1]).tocoo()
            ident.col = ident.col[new_order]
            new_mat = new_mat.dot(ident)
        return new_mat

    def get_lu(self):
        return linalg.lu(self.a_d.toarray(), permute_l=False)
        # Since these matrices have no entries equal to 0 on the main diagonal no swaps are needed
        U = self.a_d.tocsc()
        entry_list = []
        for k in range(self.a_d.shape[0]):
            for l in range(k + 1, self.a_d.shape[0]):
                mul = U[l, k] / U[k, k]
                entry_list += [(k, l, mul)]
                U = add_row_to_row(U, l, k, -mul)


    def eval_sparsity_lu(self):
        lu = self.get_lu()
        sub = sparse.diags([-1], [0], shape = ((self.n-1)**self.d, (self.n-1)**self.d))
        result = lu[0] + lu[1] + sub
        abs_non_zero = result.count_nonzero()
        abs_entries = ((self.n-1)**self.d)**2
        rel_non_zero = abs_non_zero / abs_entries
        return abs_non_zero, rel_non_zero

    def graph():
        pass

def add_row_to_row(mat, a, b, value = 1):
    new_mat = mat
    ident = sparse.eye(mat.shape[0]).tolil()
    ident[a,b]=value
    new_mat = ident.dot(new_mat)
    return new_mat

def main():
    '''Example of code that can be run using the provided class and methods
    '''
    mat_1 = BlockMatrix(2, 3)
#
    #print(mat_1.get_sparse().toarray())
    #print(mat_1.eval_sparsity())
#
    #mat_2 = BlockMatrix(3, 5)
#
    #print(mat_2.get_sparse().toarray())
    #print(mat_2.eval_sparsity())

    lu = mat_1.get_lu()
    #print(lu[0], "\n", lu[1])
    sparsity_lu = mat_1.eval_sparsity_lu()
    #print(sparsity_lu)

    #swap = mat_2.sparse_swapper(mat_2.get_sparse(), 0, 1, "row")
    #swap = swap.toarray()
    #print(swap)
#
    #add = add_row_to_row(mat_2.get_sparse(), 0, 1)
    #add = add.toarray()
    #print(add)


if __name__ == "__main__":
    main()

# Der Datentyp von a_d ist scipy.sparse._csr.csr_array und der Test verlangt etwas anderes.
# wir konnten das geforderte format nicht erzeugen aber meinen, dass dieses gleichbedeutend sein
# sollte.