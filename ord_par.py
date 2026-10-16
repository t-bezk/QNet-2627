from uv_network import gen_uv_flower, add_weights
import numpy as np
from sm_trans import build, reduce_to_AB
import matplotlib.pyplot as plt

G = gen_uv_flower(5)

p = np.linspace(0.2,0.8,50)

P_inf = []

for _p in p:
    df = build(add_weights(G, _p))

    P_inf.append(reduce_to_AB(df, 1, 2))

plt.scatter(p,P_inf)
plt.show()
