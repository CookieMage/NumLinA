import numpy as np
from scipy import sparse

class BlockMatrix:
    d = None # int
    n = None # int
    a_1 = None # coo_array
    a_2 = None # coo_array
    a_3 = None # coo_array
    def __init__(self, d, n):
        if d < 1 or d > 3:
            raise ValueError
        self.d = d
        self.n = n
        self.a_1 = sparse.diags([-1, 2*self.d, -1], [-1, 0, 1], shape=(self.n-1, self.n-1), format = "coo")
    def get_sparse(self):
        print("No")
        # data = np.array([2*self.d, -1] + [0 for _ in range(2, self.n)])
        #offset = np.array([])
        #a_1 = sparse.diags([-1, 2*self.d, -1], [-1, 0, 1], shape=(self.n-1, self.n-1)).toarray()
        #print(a_1)
        #print(-1*sparse.identity((self.n-1)**1), a_1, -1*sparse.identity((self.n-1)**1))
        #a_1_id = sparse.hstack(-1*sparse.identity((self.n-1)**1), a_1)#, -1*sparse.identity((self.n-1)**1))
        #print(a_1_id)
        #a_2 = sparse.block_diag((a_1_id)).toarray()
        #print(a_2)
    
    def eval_sparsity(self):
        abs_non_zero = self.a_1.count_nonzero()
        abs = (self.n-1)**2
        rel_non_zero = abs_non_zero / abs
        return abs_non_zero, rel_non_zero
    
block = BlockMatrix(2, 10)
print(block.a_1.toarray())
print(block.eval_sparsity())