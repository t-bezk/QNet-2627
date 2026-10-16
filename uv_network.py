""" Generates an Nth generation (2,2) flower """

import numpy as np

def gen_uv_flower(n0_gens):
    uv_arr = [(0,1)]
    cnt = 2

    for _ in range(n0_gens):
        old_g = uv_arr.copy()

        for i, _arr in enumerate(old_g):
            a = _arr[0]
            b = _arr[1]
            uv_arr.pop(i)
            uv_arr += [(a, cnt),(cnt, b),(a,cnt+1),(cnt+1, b)]
            cnt+=2
    return uv_arr


def add_weights(graph_edges, x):
    """Add an homogeneous weight x column to an array of connected nodes
    Args:
        graph_edges: array of edges (N,2)
        x: parameter value across network
    Returns:
        (N,3) array
    """
    return np.column_stack((graph_edges, np.full(len(graph_edges), x)))
