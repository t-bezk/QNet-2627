import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def seri(t1,t2):
    return t1*t2

def para(t1,t2):
    return 1 - (1-t1)*(1-t2)


G = nx.Graph()

#G.add_nodes_from([0,1,2,3])
#G.add_edges_from([(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)])

G.add_nodes_from([0,1,2,3,4])
G.add_edges_from([(0,1),(0,2),(0,3),(0,4)])

nx.draw_kamada_kawai(G, with_labels=True)
plt.show()

