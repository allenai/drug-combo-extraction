'''
Run test with `pytest`
'''

from collections import Counter
import pytest

from .utils import is_sublist

def test_is_sublist():
    superlist = ["a", "b", "c", "d"]
    sublist_1 = ["a", "b"]
    sublist_2 = ["b", "c", "d"]
    sublist_3 = ["d"]
    sublist_4 = []
    not_sublist_1 = ["a", "b", "c", "d", "e"]
    not_sublist_2 = ["a", "b", "e"]
    not_sublist_3 = ["f"]

    for sublist in [sublist_1, sublist_2, sublist_3, sublist_4]:
        assert is_sublist(sublist, superlist)

    for not_sublist in [not_sublist_1, not_sublist_2, not_sublist_3]:
        assert not is_sublist(not_sublist, superlist)
