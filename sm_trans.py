import numpy as np
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import root


def seri(t1,t2):
    return t1*t2


def para(t1,t2):
    return 1 - (1-t1)*(1-t2)


def pair(a, b):
    if a < b:
        return (a, b)
    else:
        return (b, a)


def cycle5_to_star3():
    G = nx.Graph()

    c = [0.5, 0.5, 0.5, 0.5, 0.5]

    t = {
        'c15': c[0],
        'c25': c[1],
        'c24': c[0],
        'c34': c[0],
        'c13': c[0],

    }

    G.add_nodes_from([1,2,3,4,5])
    G.add_edges_from([(1,5),(2,5),(2,4),(3,4),(1,3)])
    #nx.draw_kamada_kawai(G, with_labels=True)

    t12 = para( seri(t['c15'],t['c25']), seri(seri(t['c13'],t['c34']),t['c24']) )
    t13 = para(t['c13'], seri(seri(t['c15'],t['c25']),seri(t['c24'],t['c34'])))
    t23 = para(seri(t['c24'],t['c34']), seri(seri(t['c13'],t['c15']),t['c25']))
    print(f'{t12}, {t13}, {t23}')


def cross(i, j, W, nodes):
    nodes = list(nodes)
    if len(nodes) == 2:
        return W[pair(i, j)]
    k = next(n for n in nodes if n not in (i, j))
    rest = [n for n in nodes if n != k]
    phi = star_to_mesh({a: W[pair(a, k)] for a in rest})

    w2 = {p: para(W[p], phi[p]) for p in phi}
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
    sol = root(residuals, x0, method="hybr", tol=1e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    return dict(zip(pairs, sol.x))



def reduce_to_AB(G, A, B):
    H = G.copy()
    while True:
        others = [n for n in H.nodes if n not in (A, B)]
        if not others:
            break
        k = min(others, key=H.degree)
        nbrs = list(H.neighbors(k))
        if len(nbrs) >= 2:
            legs = {a: H[a][k]["weight"] for a in nbrs}
            for (a, b), w in star_to_mesh(legs).items():
                if H.has_edge(a, b):
                    H[a][b]["weight"] = para(H[a][b]["weight"], w)
                else:
                    H.add_edge(a, b, weight=w)
        H.remove_node(k)
    return H[A][B]["weight"] if H.has_edge(A, B) else 0.0


def exact_AB(G, A, B):
    edges = list(G.edges(data="weight"))
    total = 0.0
    for state in itertools.product([0, 1], repeat=len(edges)):
        prob, open_edges = 1.0, []
        for (u, v, p), s in zip(edges, state):
            prob *= p if s else 1 - p
            if s:
                open_edges.append((u, v))
        H = nx.Graph(open_edges)
        H.add_nodes_from([A, B])
        if nx.has_path(H, A, B):
            total += prob
    return total


def build(edge_list):
    G = nx.Graph()
    G.add_weighted_edges_from(edge_list)
    return G
