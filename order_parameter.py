import numpy as np
import perc_models as pm
import sm_in3 as sm

def route_matrix(n, i):

    if n < 2:
        raise ValueError("n must be >= 2")
    if not 0 <= i < 2 ** (n - 1):
        raise IndexError(f"i must satisfy 0 <= i < {2 ** (n - 1)}")

    N = n + 2
    A = np.zeros((N, N), dtype=int)

    for k in range(2, N + 1):
        A[k - 2, k - 1] = A[k - 1, k - 2] = 1

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

def locate_new_edges(n, i):

    ad = np.array(route_matrix(n-1,i+1))
    ad_2 = np.array(get_reduced_network(n-1,i+1))

    ## all points considered in fwd star mesh are in this row
    sm_pts = np.ravel(np.where(np.array(ad[-2] == 1)))

    ad = np.delete(ad, -2, 0)
    ad = np.delete(ad, -2, 1)

    ad_mask = [[1 if ad[i][j] != ad_2[i][j]
                else 0 for j in range(len(ad[0]))] for i in range(len(ad))]

    ## common edge with no extra parallel connectivity
    m_pt = np.argwhere(np.array(ad_mask) == 1)[0]

    sd_vl = np.ravel(np.intersect1d(m_pt, sm_pts))

    return np.concatenate([sd_vl, np.delete(sm_pts, np.where(sm_pts == sd_vl))])

def get_weighted_route(n, i, p):
    """Weighted adjacency matrix of an n-node graph, route 1 <= i <= 2**(n-3)

    p = [p0, p1, ..., p_{n-2}]  (n-1 weights); needs n >= 4.
    Relies on route_matrix(n_old, idx) from before, with n_old = n-2, idx = i-1.
    """
    p = np.asarray(p, dtype=float)
    if len(p) < n - 1:
        raise ValueError(f"need at least n-1 = {n-1} weights in p")
    if not 1 <= i <= 2 ** (n - 3):
        raise IndexError(f"i must satisfy 1 <= i <= {2 ** (n - 3)}")

    A = np.triu(route_matrix(n - 2, i - 1), 1)
    W = np.zeros((n, n))

    prev = np.flatnonzero(A[:, n - 1])
    W[prev, n - 1] = p[0]

    for c in range(n - 2, 0, -1):
        k = n - 1 - c
        rows = np.flatnonzero(A[:, c])
        if c == 1:
            W[rows, c] = p[k - 1] ** 2
        else:
            shared = np.intersect1d(rows, prev)
            other = np.setdiff1d(rows, prev)
            W[shared, c] = p[k - 1] ** 2
            W[other, c] = p[k]
        prev = rows

    return W + W.T

def get_sc_array(p, n):
    return [ pm.P_SC(p, _n) for _n in range(n+1) ]

def eliminate_penultimate(W, E):
    """Remove node m-2 of an m-node graph"""

    m = W.shape[0]
    x, L = m - 2, m - 1

    nbrs = [v for v in np.flatnonzero(E[x]) if v != L]
    if len(nbrs) != 2 or not E[x, L]:
        raise ValueError(f"node {x} of {m}: neighbours {np.flatnonzero(E[x]).tolist()}")

    nbrs.sort(key=lambda v: not E[v, L])    ## neighbour already joined to L first
    s, o = nbrs

    d, e, f = sm.star_3_fwd(W[x, s], W[x, L], W[x, o])     ## a=s, b=L, c=o

    R, F = W.copy(), E.copy()
    for (u, v), w in (((s, L), f), ((s, o), d), ((o, L), e)):
        R[u, v] = R[v, u] = pm.para(W[u, v], w) if E[u, v] else w
        F[u, v] = F[v, u] = True

    keep = [j for j in range(m) if j != x]
    return R[np.ix_(keep, keep)], F[np.ix_(keep, keep)]

def repeat_star_to_3(n, i, p):
    """"""

    E = route_matrix(n - 2, i - 1).astype(bool)   ## edge pattern from the Kronecker structure
    W = get_weighted_route(n, i, p)
    while W.shape[0] > 3:
        W, E = eliminate_penultimate(W, E)
    return W


def get_node_param(g, p):
    """Order parameter and sponge crossing of the g-th generation (2,2) flower at link prob p"""
    if g < 1:
        raise ValueError("g must be >= 1")
    n = g + 2                                     # number of nodes in the reduced network
    i_max = 2 ** (g - 1)
    p_inf_i = 0.
    p_sc_i = 0.

    for i in range(1, i_max + 1):
        if g == 1:
            TRI = np.array([[0, p**2, p], [p**2, 0, p], [p, p, 0]])
        else:
            TRI = repeat_star_to_3(n, i, get_sc_array(p, g))

        D_AB, F_TA, E_TB = TRI[0][1], TRI[0][2], TRI[1][2]

        AM, TM, BM = sm.star_3_inv(D_AB, E_TB, F_TA)

        p_inf_i += 0.5 * (TM * AM + TM * BM) / i_max
        p_sc_i += AM * BM / i_max

    return p_inf_i, p_sc_i

