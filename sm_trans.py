import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import fsolve


def seri(t1,t2):
    return t1*t2

def para(t1,t2):
    return 1 - (1-t1)*(1-t2)


#G = nx.Graph()
#G.add_nodes_from([0,1,2,3,4])
#G.add_edges_from([(0,1),(0,2),(0,3),(0,4)])
#
#nx.draw_kamada_kawai(G, with_labels=True)
#plt.show()


## 5 cycle to 3 branch star
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
nx.draw_kamada_kawai(G, with_labels=True)
#plt.show()

t12 = para( seri(t['c15'],t['c25']), seri(seri(t['c13'],t['c34']),t['c24']) )
t13 = para(t['c13'], seri(seri(t['c15'],t['c25']),seri(t['c24'],t['c34'])))
t23 = para(seri(t['c24'],t['c34']), seri(seri(t['c13'],t['c15']),t['c25']))

print(f'{t12}, {t13}, {t23}')
