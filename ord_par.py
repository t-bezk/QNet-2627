from UVnetwork import gen_uv_flower_format
import numpy as np
from sm_trans import build, reduce_to_AB
import matplotlib.pyplot as plt

G = gen_uv_flower_format(5)

p = np.linspace(0.55,0.65,50)

P_inf = []

def network_resort(graph, p):
    return np.column_stack((graph, np.full(len(graph), p)))

for __p in p:
    df = build(network_resort(G, __p))

    P_inf.append(reduce_to_AB(df, 1, 2))

plt.scatter(p,P_inf)
plt.show()
