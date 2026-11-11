import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def route_matrix(n, i):

    if n < 2:
        raise ValueError("n must be >= 2")
    if not 0 <= i < 2 ** (n - 1):
        raise IndexError(f"i must satisfy 0 <= i < {2 ** (n - 1)}")

    N = n + 2
    A = np.zeros((N, N), dtype=int)

    # path part: A[k-1, k] = 1 (1-indexed), plus mirror
    for k in range(2, N + 1):
        A[k - 2, k - 1] = A[k - 1, k - 2] = 1

    # second 1 of each column k = 3..N
    p = 1
    A[p - 1, 2] = A[2, p - 1] = 1
    for t in range(1, n):
        k = t + 3
        s = (i >> (n - 1 - t)) & 1
        if s == 1:
            p = k - 2
        A[p - 1, k - 1] = A[k - 1, p - 1] = 1
    return A

def all_matrices(n):
    return [route_matrix(n, i) for i in range(2 ** (n - 1))]

def get_reduced_network(n, i):
    return route_matrix(n-1, int(np.floor(i*0.5)))

def locate_new_edge(n, i):

    ad = np.array(route_matrix(n,i))
    ad_2 = np.array(get_reduced_network(n,i))

    ## all points considered in fwd star mesh are in this row
    sm_pts = np.where(np.array(ad[-2] == 1))

    ad = np.delete(ad, -2, 0)
    ad = np.delete(ad, -2, 1)

    ad_mask = [[1 if ad[i][j] != ad_2[i][j]
                else 0 for j in range(len(ad[0]))] for i in range(len(ad))]

    ## common edge with no extra parallel connectivity
    m_pt = np.argwhere(np.array(ad_mask) == 1)

    return m_pt

n, i = 3, 1

locate_new_edge(n,i)

plt.show()
