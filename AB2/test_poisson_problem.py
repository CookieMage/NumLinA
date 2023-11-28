import itertools
import inspect
import numpy as np
import pytest

from poisson_problem import rhs, idx, inv_idx

# pylint: disable=missing-docstring
# pylint: disable=invalid-name

# Define some functions
def identity(x1):
    return x1

def c1(x1):
    return 1

def c2(x1, x2):
    return 1

def c3(x1, x2, x3):
    return 1

def f1(x1):
    return x1**2

def g1(x1):
    return np.sin(x1)

def f2(x1, x2):
    j = 1
    return - 2*j*np.pi*(x1*np.sin(np.pi*j*x1)*np.cos(np.pi*j*x2)
                        + x2*np.sin(np.pi*j*x2)
                        * (np.cos(np.pi*j*x1) - np.pi*j*x1*np.sin(np.pi*j*x1)))

def g2(x1, x2):
    return np.exp(x1) + x2

def f3(x1, x2, x3):
    return x1 + x2 * x3

def g3(x1, x2, x3):
    return x1**2 + x2 + x3**3

# Prepare for different signatures of f
def wrapped_rhs(d, n, f):
    try:
        return rhs(d, n, lambda x: f(*x))
    except TypeError:
        return rhs(d, n, f)


TOL = 10e-14
N_LIST = [2, 3, 7, 10]
F_LIST = [identity, c1, c2, c3, f1, g1, f2, g2, f3, g3]

# test ValueError for impossible d
@pytest.mark.parametrize('d, n, f', itertools.product([-3, -1, 0], N_LIST, F_LIST))
def test_value_error_for_d(d, n, f):
    with pytest.raises(ValueError):
        wrapped_rhs(d, n, f)


@pytest.mark.parametrize('n, f', itertools.product([-1, 0, 1], F_LIST))
def test_value_error_for_n(n, f):
    #Find out d from signature
    d = len(inspect.signature(f).parameters)
    # Check ValueError for impossible n
    with pytest.raises(ValueError):
        wrapped_rhs(d, n, f)

# Test some solutions
@pytest.mark.parametrize('n, f', itertools.product(N_LIST, F_LIST))
def test_values(n, f):
    d = len(inspect.signature(f).parameters)
    filename = "reference_files/" + str(d) + str(n) + str(f)[10:12] + "rhs.npy"
    ref = np.load(filename)
    sol = wrapped_rhs(d, n, f)
    assert max(abs(sol - ref)) < TOL

# Test some idx and inv_idx
def test_idx_d1():
    check = True
    for i in range(1, 7):
        check = check and (i == idx([i], 8))
    assert check

def test_inv_idx_d1():
    check = True
    for i in range(1, 7):
        check = check and ([i] == inv_idx(i, 1, 8))
    assert check

X = [[1, 1], [3, 1], [1, 2], [3, 2], [1, 3], [3, 3], [1, 4]]

def test_idx_d2():
    check = True
    for i in range(7):
        check = check and (2*i+1 == idx(X[i], 5))
    assert check

def test_inv_idx_d2():
    check = True
    for i in range(7):
        check = check and (X[i] == inv_idx(2*i+1, 2, 5))
    assert check

Y = [[1, 1, 1], [1, 2, 1], [1, 3, 1], [1, 1, 2], [1, 2, 2], [1, 3, 2], [1, 1, 3], [1, 2, 3]]

def test_idx_d3():
    check = True
    for i in range(8):
        check = check and (3*i+1 == idx(Y[i], 4))
    assert check

def test_inv_idx_d3():
    check = True
    for i in range(8):
        check = check and (Y[i] == inv_idx(3*i+1, 3, 4))
    assert check
