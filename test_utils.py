
'''
Run test with `pytest`
'''
import pytest

from utils import tuple_difference, tuple_max

def test_tuple_difference():
    tuple_a = (1, 2, 3)
    tuple_b = (2, 3, 4)
    diff_tuple = tuple_difference(tuple_a, tuple_b)
    assert diff_tuple == (1, 1, 1)

def test_tuple_difference_invalid():
    tuple_a = (1, 2, 3)
    tuple_b = (0, 1, 2)
    with pytest.raises(AssertionError):
        tuple_difference(tuple_a, tuple_b)

def test_tuple_max():
    tuple_a = (1, 2, 3)
    tuple_b = (3, 2, 1)
    diff_tuple = tuple_max(tuple_a, tuple_b)
    assert diff_tuple == (3, 2, 3)

def test_tuple_max_when_tuple_is_None():
    tuple_a = None
    tuple_b = (0, 1, 2)
    diff_tuple = tuple_max(tuple_a, tuple_b)
    assert diff_tuple == tuple_b
    diff_tuple = tuple_max(tuple_b, tuple_a)
    assert diff_tuple == tuple_b

