import numpy as np
from scipy import sparse

class BlockMatrix:
    d = None
    n = None
    def __init__(self, d, n):
        if d < 1 or d > 3:
            raise ValueError
        self.d = d
        self.n = n
    def get_sparse(self):
        data = np.array([2*self.d, -1] + [0 for _ in range(2, self.n)])
        offset = np.array([])
        a_1 = sparse.diags([-1, 2*self.d, -1], [-1, 0, 1], shape=(self.n-1, self.n-1)).toarray()
        print(a_1)
        a_2 = sparse.block_diag(([-1*sparse.identity((self.n-1)**1), a_1, -1*sparse.identity((self.n-1)**1)], [0])).toarray()
        print(a_2)
    def eval_sparsity(self):
        abs_non_zero = self.sparse.dia_array.count_nonzero()
        abs = (self.n-1)**2
        rel_non_zero = abs_non_zero / abs
        return abs_non_zero, rel_non_zero
    
block = BlockMatrix(2, 5)
block.get_sparse()