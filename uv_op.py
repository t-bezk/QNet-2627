import numpy as np
from random import random
import matplotlib.pyplot as plt

TRIALS = 1000
GR_RES = 100


def n1_22_flower(p,p_inf):
    for q in range(GR_RES):
        for _ in range(TRIALS):
            r = [ random() for _ in range(4) ]
            adj_m = [ 1 if r[i] < p[q] else 0 for i in range(4) ]

            n_conn = (adj_m[0] or adj_m[1]) + (adj_m[2] or adj_m[3])
            p_inf[q] += n_conn / 2



pr = np.linspace(0,1,GR_RES)

op_inf = np.zeros(GR_RES)


n1_22_flower(pr,op_inf)
plt.scatter(pr,op_inf/np.max(op_inf))
plt.plot(pr,1-(1-pr)**2)

plt.show()

import itertools
import random

import numpy as np

import sm_trans as sm

T, A, B = 0, 1, 2


def reduce_keep(G, keep):
    H = {n: dict(nb) for n, nb in G.items()}
    while True:
        others = [n for n in H if n not in keep]
        if not others:
            return H
        k = min(others, key=lambda n: len(H[n]))
        nbrs = list(H[k])
        if len(nbrs) >= 2:
            for (a, b), w in sm.star_to_mesh_k3({a: H[k][a] for a in nbrs}).items():
                if b in H[a]:
                    H[a][b] = H[b][a] = sm.para(H[a][b], w)
                else:
                    H[a][b] = H[b][a] = w
        for a in nbrs:
            del H[a][k]
        del H[k]


def layer_step(state, w, d, U, V):
    wtu, wtv, wuv = state
    L, Lo = (U, V) if d < U else (V, U)       # path holding the ancestry link, other path
    j = d if d < U else d - U                  # link index along that path
    cj = A if j == 0 else 3                    # left end of the ancestry link
    cj1 = B if j + 1 == L else 4               # right end
    edges = [(T, cj, wtu), (T, cj1, wtv), (cj, cj1, wuv),
             (A, B, w ** Lo)]                  # the other path, collapsed in series
    if j > 0:
        edges.append((cj, A, w ** j))          # rest of this path, to the left
    if j + 1 < L:
        edges.append((cj1, B, w ** (L - j - 1)))   # ... and to the right
    H = reduce_keep(sm.build(edges), {T, A, B})
    return (H[T].get(A, 0.0), H[T].get(B, 0.0), H[A].get(B, 0.0))


def node_value(p, U, V, n, digits, pos, mode="A"):
    R = lambda x: 1 - (1 - x ** U) * (1 - x ** V)
    L, Lo = (U, V) if pos[0] == 0 else (V, U)
    state = (p ** pos[1], p ** (L - pos[1]), p ** Lo)      # t's own motif: exact triangle
    for g in range(n - 1, 0, -1):                           # parent motifs, bottom-up
        w = p
        for _ in range(n - g):
            w = R(w)
        state = layer_step(state, w, digits[g - 1], U, V)
    wtA, wtB, wAB = state
    return sm.para(wtA, wtB)                                # t reaches A or B (stop at boundary)


def pinf(p, U, V, n, mode="A", seed=0):

    positions = [(0, i) for i in range(1, U)] + [(1, i) for i in range(1, V)]
    items = ((dg, ps) for dg in itertools.product(range(U + V), repeat=n - 1) for ps in positions)


    ## Avg over all values for P_inf
    return np.mean(np.array([node_value(p, U, V, n, dg, ps, mode) for dg, ps in items]))


p_inf = [pinf(p,2,2,8) for p in pr]

plt.scatter(pr,p_inf)
plt.show()
