""" Generates an Nth generation (2,2) flower """

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

N=4

def recover_branch_layers(G):
    arr = []
    adj_mat = nx.adjacency_matrix(G).toarray()
    n = len(adj_mat)

    for i in range(n):
        for j in range(i + 1, n):  # only the entries above the diagonal
            if adj_mat[i][j] == 1:
                arr.append([i + 1, j + 1])

    return arr


def branch_out(G,u,v):
    no = G.number_of_nodes()+1
    x,y = no,no+1

    G.remove_edge(u,v)
    G.add_nodes_from([x,y])
    G.add_edges_from([(u,x),(u,y),(x,v),(y,v)])


G = nx.Graph()

## Initialise N=0 case
G.add_nodes_from([1,2])
G.add_edges_from([(1,2)])


for _ in range(N):
    uq_edge = recover_branch_layers(G)
    for r in uq_edge:
        branch_out(G, r[0],r[1])


## Display
A = nx.adjacency_matrix(G).toarray()

pos = {
    1: (-2, 0),
    2: (2, 0),
}

pos = nx.spring_layout(G, pos=pos, fixed=[1,2], seed=42)
nx.draw_kamada_kawai(G, with_labels=True, font_weight='bold')
plt.show()
