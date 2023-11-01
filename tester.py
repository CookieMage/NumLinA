import types
import pytest
import numpy as np

from derivative_approximation import  FiniteDifference as FiniteDifference_solution

# pylint: disable=missing-docstring
# pylint: disable=invalid-name

TOL = 1e-14


def f(x):
    return np.cos(x)


def d_f(x):
    return -np.sin(x)


def dd_f(x):
    return -np.cos(x)


def quad_func(x):
    return 1/2*x**2


def cubic_func(x):
    return 1/6*x**3


def biquad_fucn(x):
    return 1/24*x**4


def test_signature_fd():
    fin_diff = FiniteDifference_solution(1, f, d_f, dd_f)
    assert isinstance(fin_diff.compute_dh_f(), types.FunctionType) and \
        isinstance(fin_diff.compute_ddh_f(), types.FunctionType)


def test_signature_compute_errors():
    fin_diff = FiniteDifference_solution(1, f, d_f, dd_f)
    err1, err2 = fin_diff.compute_errors(1, 2, 10)
    assert isinstance(err1, float) and isinstance(err2, float)


def test_error_sign():
    fin_diff_p = FiniteDifference_solution(1, biquad_fucn,
                                           cubic_func, quad_func)
    err1_p, err2_p = fin_diff_p.compute_errors(1, 2, 2)
    fin_diff_n = FiniteDifference_solution(1, lambda x: -biquad_fucn(x),
                                           lambda x: -cubic_func(x),
                                           lambda x: -quad_func(x))
    err1_n, err2_n = fin_diff_n.compute_errors(1, 2, 2)
    print("err1_n: ", err1_n)
    print("err2_n: ", err2_n)
    print("err1_p: ", err1_p)
    print("err2_p: ", err2_p)
    assert err1_p >= 0 and err2_p >= 0 and err1_n >= 0 and err2_n >= 0


def test_valueerror():
    with pytest.raises(ValueError):
        fin_diff = FiniteDifference_solution(0.001, f)
        fin_diff.compute_errors(-np.pi, np.pi, 1000)
        
test_error_sign()
test_signature_compute_errors()
test_signature_fd()
test_valueerror()