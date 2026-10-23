import numpy as np
from random import random
import matplotlib.pyplot as plt

NO_TRS = 4
TRIALS = 10000
GR_RES = 100

p = np.linspace(0,1,GR_RES)


p_inf = np.zeros(GR_RES)

for q in range(GR_RES):
    for _ in range(TRIALS):
        r = [ random() for _ in range(NO_TRS) ]
        adj_m = [ 1 if r[i] < p[q] else 0 for i in range(NO_TRS) ]

        ## for n=1 flower case
        if adj_m[0] or adj_m[1] or adj_m[2] or adj_m[3]:
            p_inf[q]+=1

plt.scatter(p,p_inf/np.max(p_inf))
plt.plot(p, 2*p-p**2)
plt.show()
