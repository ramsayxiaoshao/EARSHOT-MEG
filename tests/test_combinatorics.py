from itertools import combinations

import pytest

from earshot.combinatorics import combination, n_possibilities


@pytest.mark.parametrize("start, stop, n_sample", [
    (1,  4, 2),
    (1, 15, 6),
    (2, 10, 5),
    (8, 15, 3),
    (8, 15, 7),
])
def test_n_possibilities(start, stop, n_sample):
    assert n_possibilities(start, stop, n_sample) == len(list(combinations(range(start, stop), n_sample)))


@pytest.mark.parametrize("stop, n_sample, i", [
    (10, 3, 10),
    (10, 5, 12),
    (8,  4,  1),
    (8,  4, 30),
])
def test_combination(stop, n_sample, i):
    assert combination(stop, n_sample, i) == list(list(combinations(range(stop), n_sample))[i])
