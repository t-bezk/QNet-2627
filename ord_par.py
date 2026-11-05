from uv_network import gen_uv_flower, add_weights
import numpy as np
from sm_trans import build, reduce_to_AB
import matplotlib.pyplot as plt
from random import randint
from sm_trans import *
from sm_in3 import star_3_fwd, star_3_inv

def n_nodes(n_gens):
    return (4 + 2 * 4 ** n_gens) // 3


def sm_gen(p, n0_gens):
    graph_net = gen_uv_flower(n0_gens)
    repack_p = build(add_weights(graph_net, p))
    return reduce_to_AB(repack_p, 1, 2)

def pinf_gen(p, n0_gens):
    k=0.1
    graph_net = gen_uv_flower(n0_gens)
    repack_p = build(add_weights(graph_net, p))
    no_set = []
    for _ in range(int(np.ceil(len(repack_p)*k))):
        no_set.append(randint(3,n_nodes(n0_gens)))
    return np.mean([reduce_to_AB(repack_p, t, 2) for t in np.array(no_set)])


RNG = 5001

pr=np.linspace(0,0.84,RNG)

P_INF_N = []

for p in pr:
    AX = para(seri(p,p),seri(p,p))
    XT = p
    BT = p
    BX = seri(p,p)
    AB = para(para(seri(p,p),seri(p,p)),para(seri(p,p),seri(p,p)))

    D, E, F = star_3_fwd(BX, XT, AX)

    BT_2 = para(F,BT)
    AB_2 = para(D,AB)

    AT, BT, AB_3 = star_3_inv(E, BT_2, AB_2)

    P_INF_N.append(0.5 * (AT + BT))

plt.scatter(pr,P_INF_N)
plt.scatter(pr,para(para(seri(pr,pr),seri(pr,pr)),para(seri(pr,pr),seri(pr,pr))))
plt.show()
