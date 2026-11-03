import numpy as np
import itertools
import matplotlib.pyplot as plt
from scipy.optimize import root, brentq
import timeit


def seri(t1,t2):
    return t1*t2


def para(t1,t2):
    return 1 - (1-t1)*(1-t2)


def pair(a, b):
    if a < b:
        return (a, b)
    else:
        return (b, a)


def cross(i, j, w, nodes):
    nodes = list(nodes)
    if len(nodes) == 2:
        return w[pair(i, j)]
    k = next(n for n in nodes if n not in (i, j))
    rest = [n for n in nodes if n != k]
    phi = star_to_mesh({a: w[pair(a, k)] for a in rest})

    w2 = {p: para(w[p], phi[p]) for p in phi}
    return cross(i, j, w2, rest)


def star_to_mesh(legs):
    leaves = sorted(legs)
    if len(leaves) == 2:
        a, b = leaves
        return {(a, b): seri(legs[a], legs[b])}
    pairs = list(itertools.combinations(leaves, 2))

    def residuals(x):
        W = dict(zip(pairs, x))
        return [cross(i, j, W, leaves) - seri(legs[i], legs[j]) for i, j in pairs]

    x0 = [0.5 * seri(legs[i], legs[j]) for i, j in pairs]
    sol = root(residuals, x0, method="hybr", tol=1e-8)
    return dict(zip(pairs, sol.x))


def reduce_to_AB(G, A, B):

    time_st = timeit.default_timer()

    H = {n: dict(nbrs) for n, nbrs in G.items()}
    while True:
        others = [n for n in H if n not in (A, B)]
        if not others:
            break
        k = min(others, key=lambda n: len(H[n]))
        nbrs = list(H[k])
        if len(nbrs) >= 2:
            legs = {a: H[k][a] for a in nbrs}
            for (a, b), w in star_to_mesh(legs).items():
                if b in H[a]:
                    H[a][b] = H[b][a] = para(H[a][b], w)
                else:
                    H[a][b] = H[b][a] = w
        for a in nbrs:
            del H[a][k]
        del H[k]

    time_en = timeit.default_timer()

    print(f'SM completed in {round((time_en - time_st), 3)}s')

    return H.get(A, {}).get(B, 0.0)
 
 
def exact_AB(G, A, B):
    edges, done = [], set()
    for u in G:
        for v, w in G[u].items():
            if v not in done:
                edges.append((u, v, w))
        done.add(u)

    total = 0.0
    for state in itertools.product([0, 1], repeat=len(edges)):
        prob, adj = 1.0, {}
        for (u, v, p), s in zip(edges, state):
            prob *= p if s else 1 - p
            if s:
                adj.setdefault(u, []).append(v)
                adj.setdefault(v, []).append(u)
        seen, stack = {A}, [A]
        while stack:
            n = stack.pop()
            for m in adj.get(n, ()):
                if m not in seen:
                    seen.add(m)
                    stack.append(m)
        if B in seen:
            total += prob
    return total


def build(edge_list):
    graph_obj = {}
    for u, v, w in edge_list:
        graph_obj.setdefault(u, {})[v] = w
        graph_obj.setdefault(v, {})[u] = w
    return graph_obj


def _star_to_triangle(legs):

    i, j, k = sorted(legs)
    a_ij, a_ik, a_jk = (1 - legs[i] * legs[j], 1 - legs[i] * legs[k], 1 - legs[j] * legs[k])
    c1, c2, c3 = a_ij + a_ik - a_jk, a_ij + a_jk - a_ik, a_ik + a_jk - a_ij
 
    def g(x):
        return (x + c1) * (x + c2) * (x + c3) - 8 * x * x   # g(0) >= 0 >= g(1)
 
    if g(0.0) <= 0:                      # guard against rounding at the endpoints
        S = 0.0
    elif g(1.0) >= 0:
        S = 1.0
    else:
        S = brentq(g, 0.0, 1.0, xtol=1e-15)
    u, v, w = (c1 + S) / 2, (c2 + S) / 2, (c3 + S) / 2
 
    def q(x, y, z):
        return np.sqrt(x * y / z) if z > 0 else 0.0
 
    return {(i, j): 1 - q(u, v, w), (i, k): 1 - q(u, w, v), (j, k): 1 - q(v, w, u)}
 
 
def star_to_mesh_k3(legs):
    leaves = sorted(legs)
    if len(leaves) == 2:
        a, b = leaves
        return {(a, b): seri(legs[a], legs[b])}
    if len(leaves) == 3:                         # closed form: no solver needed
        return _star_to_triangle(legs)
    pairs = list(itertools.combinations(leaves, 2))
 
    def residuals(x):
        W = dict(zip(pairs, x))
        return [cross(i, j, W, leaves) - seri(legs[i], legs[j]) for i, j in pairs]
 
    x0 = [0.5 * seri(legs[i], legs[j]) for i, j in pairs]
    sol = root(residuals, x0, method="hybr", tol=1e-13)
    # Judge convergence by the residual, not sol.success: with a tight tol the flag
    # can be False even when the equations are already solved to ~1e-16.
    if not sol.success and max(abs(r) for r in residuals(sol.x)) > 1e-10:
        raise RuntimeError(sol.message)
    return dict(zip(pairs, sol.x))
