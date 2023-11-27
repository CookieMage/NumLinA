import numpy as np
from scipy import sparse

class BlockMatrix:
    d = None # int
    n = None # int
    a_d = None # csr_array

    def __init__(self, d : int, n : int):
        if not isinstance(d, int) or not isinstance(n, int):
            raise TypeError
        if d < 1 or d > 3 or n < 2:
            raise ValueError
        self.d = d
        self.n = n
        a_1 = sparse.diags([-1, 2*self.d, -1], [-1, 0, 1], shape=(self.n-1, self.n-1))
        if d == 1:
            self.a_d = a_1
        else:
            # generiere blockmatrix, welche a_1 enthält
            a_2_block = sparse.block_diag([a_1 for _ in range(self.n-1)])
            # generiere Blockmatrix, welche die Identitätsmatrizen enthält
            a_2_ident = sparse.diags([-1, -1], [-(self.n-1), self.n-1], shape=((self.n-1)**2, (self.n-1)**2))
            # addiere die Matrizen, um die gesuchte Matrix a_2 zu erhalten
            a_2 = a_2_block + a_2_ident
            if d == 2:
                self.a_d = a_2
            else:
                # generiere blockmatrix, welche a_2 enthält
                a_3_block = sparse.block_diag([a_2 for _ in range(self.n-1)])
                # generiere Blockmatrix, welche die Identitätsmatrizen enthält
                a_3_ident = sparse.diags([-1, -1], [-(self.n-1)**2, (self.n-1)**2], shape=((self.n-1)**3, (self.n-1)**3))
                # addiere die Matrizen, um die gesuchte Matrix a_2 zu erhalten
                self.a_d = a_3_block + a_3_ident

        # a_d_block = sparse.block_diag([a_1 for _ in range((self.n-1)**d)])
        # a_d_ident = sparse.diags([-1, -1], [-(self.n-1)**(d-1), (self.n-1)**(d-1)], shape=((self.n-1)**d, (self.n-1)**d))


    def get_sparse(self):
        return self.a_d.toarray()

    def eval_sparsity(self):
        abs_non_zero = self.a_d.count_nonzero()
        abs = ((self.n-1)**self.d)**2
        rel_non_zero = abs_non_zero / abs
        return abs_non_zero, rel_non_zero

