def add_row_to_row(mat, a, b, value = 1):
    new_mat = mat
    ident = sparse.eye(mat.shape[0]).tolil()
    ident[a,b]=value
    new_mat = ident.dot(new_mat)
    return new_mat

def sparse_swapper(mat, a, b, mode="row"):
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
    # Since these matrices have no entries equal to 0 on the main diagonal no swaps are needed
    #U = self.a_d.tocsc()
    #entry_list = []
    #for k in range(self.a_d.shape[0]):
    #    for l in range(k + 1, self.a_d.shape[0]):
    #        mul = U[l, k] / U[k, k]
    #        entry_list += [(k, l, mul)]
    #        U = add_row_to_row(U, l, k, -mul)