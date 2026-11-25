
import itertools
from collections import namedtuple
from functools import reduce
import matplotlib.pyplot as plt

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

Flower = namedtuple("Flower", "edges n_nodes first_deep U V n")
Rules = namedtuple("Rules", "seri para name")

CLASSICAL = Rules(seri=lambda a, b: a * b,
                  para=lambda a, b: 1 - (1 - a) * (1 - b),
                  name="classical")


# ---------------------------------------------------------------------------
# 1. Flower generator
# ---------------------------------------------------------------------------
def uv_flower(U, V, n):
    """n-th generation (U,V) flower.  Each generation replaces every link (u,v) by a
    U-link path and a V-link path in parallel, each with fresh interior nodes.

    Edge order matters and is guaranteed: the (U+V) children of parent link k occupy rows
    k*(U+V) ... (k+1)*(U+V)-1, the first U of them being the U-path (in order from u to v),
    the rest the V-path.  Rules can therefore be applied by reshaping (see _evaluate).
    """
    if U < 1 or V < 1 or (U == 1 and V == 1):
        raise ValueError("need U, V >= 1, not both 1")
    edges = np.array([[0, 1]], dtype=np.int64)
    N, first_deep = 2, 0                         # n = 0: just A and B
    per = U + V - 2                              # interior nodes created per parent link
    for _ in range(n):
        E = len(edges)
        u, v = edges[:, 0], edges[:, 1]
        base = N + per * np.arange(E)            # first new label for each parent link
        first_deep = N
        paths = []
        for L, offset in ((U, 0), (V, U - 1)):
            inner = base[:, None] + offset + np.arange(L - 1)[None, :]       # (E, L-1)
            seq = np.concatenate([u[:, None], inner, v[:, None]], axis=1)    # (E, L+1)
            paths.append(np.stack([seq[:, :-1], seq[:, 1:]], axis=2))        # (E, L, 2)
        edges = np.concatenate(paths, axis=1).reshape(-1, 2)                 # (E*(U+V), 2)
        N += per * E
    return Flower(edges, N, first_deep, U, V, n)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _evaluate(weights, U, V, rules):
    """Collapse link values to one A-B value per trial, one generation at a time.
    weights: (trials, (U+V)^n) in uv_flower edge order."""
    w = weights
    while w.shape[1] > 1:
        w = w.reshape(w.shape[0], -1, U + V)                 # group each parent's children
        up = reduce(rules.seri, [w[:, :, i] for i in range(U)])        # U-path in series
        vp = reduce(rules.seri, [w[:, :, U + i] for i in range(V)])    # V-path in series
        w = rules.para(up, vp)                               # the two paths in parallel
    return w[:, 0]


def _is_classical(rules):
    x = np.linspace(0, 1, 7)
    X, Y = np.meshgrid(x, x)
    return (np.allclose(rules.seri(X, Y), X * Y)
            and np.allclose(rules.para(X, Y), 1 - (1 - X) * (1 - Y)))


def sponge_crossing_exact(U, V, n, w, rules=CLASSICAL):
    """Deterministic value when every link carries weight w (no sampling): nest
    R(x) = para(seri^U(x), seri^V(x)) n times.  Exact for ANY rules."""
    for _ in range(n):
        up = reduce(rules.seri, [w] * U)
        vp = reduce(rules.seri, [w] * V)
        w = rules.para(up, vp)
    return w


# ---------------------------------------------------------------------------
# 2. Monte Carlo sponge crossing
# ---------------------------------------------------------------------------
def mc_sponge_crossing(flower, q, rules=CLASSICAL, w=1.0, trials=2000, seed=None,
                       max_cells=2e7):
    """Monte Carlo estimate of the A-B crossing value.

    Each trial keeps every link independently with probability q (value w) and kills it
    otherwise (value 0), then evaluates the sampled flower with the given rules.
      classical:   q = p, w = 1   -> mean is the sponge-crossing probability P_SC(p)
      rules-based: q = 1, w = c   -> deterministic (zero variance), equals the exact nest
      q < 1 and w = c             -> expected A-B value of a diluted network (a hook for noise)
    Dead links must be 0 and 0 must mean 'no link' under the rules (true for classical).
    Returns (mean, standard error).
    """
    rng = np.random.default_rng(seed)
    E = len(flower.edges)
    batch = max(1, int(max_cells // E))            # cap memory: trials x links floats
    out, done = [], 0
    while done < trials:
        b = min(batch, trials - done)
        weights = np.where(rng.random((b, E)) < q, w, 0.0)
        out.append(_evaluate(weights, flower.U, flower.V, rules))
        done += b
    vals = np.concatenate(out)
    return vals.mean(), vals.std(ddof=1) / np.sqrt(len(vals))


# ---------------------------------------------------------------------------
# 3. Monte Carlo order parameter
# ---------------------------------------------------------------------------
def mc_order_parameter(flower, q, rules=CLASSICAL, trials=500, mode="AB", seed=None):
    """Monte Carlo order parameter: the fraction of deepest-layer nodes connected to the
    boundary, averaged over random link configurations.
      mode "A"  : node connected to A          mode "AB": connected to A or B
    Uses true graph connectivity (scipy connected_components), so the rules must be the
    classical bond-percolation ones.  Returns (mean, standard error).
    """
    if not _is_classical(rules):
        raise NotImplementedError(
            "Node-to-boundary connectivity on a flower is not series-parallel, so it can't be "
            "evaluated from seri/para rules, and only classical rules have a link-sampling "
            "interpretation.  Other rules need the deterministic star-mesh route.")
    if flower.n < 1:
        raise ValueError("need n >= 1 (the deepest layer must differ from A and B)")
    rng = np.random.default_rng(seed)
    e, N = flower.edges, flower.n_nodes
    deep = np.arange(flower.first_deep, N)
    vals = np.empty(trials)
    for t in range(trials):
        keep = rng.random(len(e)) < q
        g = coo_matrix((np.ones(keep.sum()), (e[keep, 0], e[keep, 1])), shape=(N, N))
        _, lab = connected_components(g, directed=False)
        hit = lab[deep] == lab[0]
        if mode == "AB":
            hit |= lab[deep] == lab[1]
        vals[t] = hit.mean()
    return vals.mean(), vals.std(ddof=1) / np.sqrt(trials)
