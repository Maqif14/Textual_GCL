from numpy.linalg import norm
from math import sqrt, exp
from numba import jit
import heapq
import sys


def find_k_largest(K, candidates):

    n_candidates = []
    for did, score in enumerate(candidates[:K]):
        n_candidates.append((score, did))
    heapq.heapify(n_candidates)
    for did, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, did + K))
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    k_largest_scores = [item[0] for item in n_candidates]
    return ids, k_largest_scores