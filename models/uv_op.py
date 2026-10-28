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
