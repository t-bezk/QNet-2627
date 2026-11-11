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

pr=np.linspace(0,1,RNG)

P_INF_N = []

ati = []

bti = []

n_sub = [ 0, 1, 0 ]







for p in pr:

    ## two order 1 points
    T1A = para(p*p,p*p)
    T1B = para(p*p,p*p)
    P_SC = seri(para(p*p,p*p),para(p*p,p*p))
    AM, T1M, BM = star_3_inv(P_SC, T1A, T1B)
    P_INF_N1 = 0.5*(AM*T1M + BM*T1M)

    ## four order 2 points, rightwards chirality
    T1X = p
    T1B = p
    XB = p*p
    XA = para(p*p,p*p)
    AB_2, T1A_2, T1B_2 = star_3_fwd(XA, T1X, XB)
    AB_3 = para(P_SC, AB_2)
    T1B_3 = para(T1B_2, p)
    AM, T1M, BM = star_3_inv(AB_3, T1A_2, T1B_3)
    P_INR_N2 = 0.5*(AM*T1M + BM*T1M)

    ## four order 2 points, leftwards chirality
    T1X = p
    T1B = para(p*p,p*p)
    T1A = p*p
    XA = p
    AB_2, T1A_2, T1B_2 = star_3_fwd(T1A, T1X, T1B)
    AB_3 = para(P_SC, AB_2)
    T1B_3 = para(T1A_2, p)
    AM, T1M, BM = star_3_inv(AB_3, T1A_2, T1B_3)
    P_INL_N2 = 0.5*(AM*T1M + BM*T1M)

    ati.append((4*P_INL_N2)/4)


plt.plot(pr,ati)
#plt.scatter(pr,para(para(seri(pr,pr),seri(pr,pr)),para(seri(pr,pr),seri(pr,pr))))
plt.show()
