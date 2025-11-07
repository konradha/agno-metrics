import numpy as np


def build_pairs(N, rng):
    idx = np.arange(N)
    i = np.random.permutation(N)
    j = np.random.permutation(N)
    return i, j


def build_triplets(N, rng):
    a = np.random.permutation(N)
    p = np.random.permutation(N)
    n = np.random.permutation(N)
    return a, p, n
