"""
Humboldt-Universität zu Berlin - Projektpraktikum I
Tests block_matrix.py using the files from the folder reference_files/.

Last edited: Lara Théallier, 01.11.2021
"""
import itertools
import pytest
import scipy
import numpy
import block_matrix

# pylint: disable=missing-docstring
# pylint: disable=invalid-name

FOLDERNAME = "reference_files/"
TOL = 10e-14
DIM_LIST = [1, 2, 3]
N_LIST = [3, 4, 5, 6, 7, 8, 15]

# Check that matrix is csr
@pytest.mark.parametrize('d, n', itertools.product(DIM_LIST, N_LIST))
def test_get_sparse_type(d, n):
    sol_obj = block_matrix.BlockMatrix(d, n)
    assert isinstance(sol_obj.get_sparse(), scipy.sparse.csr_matrix)
    assert scipy.sparse.issparse(sol_obj.get_sparse())

# Check that values are correct
@pytest.mark.parametrize('d, n', itertools.product(DIM_LIST, N_LIST))
def test_get_sparse_val(d, n):
    filename = FOLDERNAME + str(d) + str(n) + "matrix.npz"
    sol_obj = block_matrix.BlockMatrix(d, n)
    ref_obj = scipy.sparse.load_npz(filename)
    assert not (sol_obj.get_sparse() - ref_obj).sum()

# Check non zeros against tolerance
@pytest.mark.parametrize('d, n', itertools.product(DIM_LIST, N_LIST))
def test_eval_sparsity(d, n):
    nnz_sol_obj = block_matrix.BlockMatrix(d, n).eval_sparsity()
    filename = FOLDERNAME + str(d) + str(n) + "zeros.npy"
    nnz_ref_obj = numpy.load(filename, fix_imports=False)
    nnz_ok = abs(nnz_sol_obj - nnz_ref_obj) < TOL
    assert nnz_ok.all()
