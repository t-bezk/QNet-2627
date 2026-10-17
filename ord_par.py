from uv_network import gen_uv_flower, add_weights
import numpy as np
from sm_trans import build, reduce_to_AB
import matplotlib.pyplot as plt
from renorm_sc_exp import *
from random import randint

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


plot_parameter(pinf_gen, 10, 6)

#p_nu_min = parameter_scaling(P_LO, pinf_gen, P_STR, DP,t='P')
#p_nu_max = parameter_scaling(P_HI, pinf_gen, P_STR, DP,t='P')
plt.show()
#print(f'beta = {0.5*(p_nu_min + p_nu_max)} ± {abs(0.5*(p_nu_min - p_nu_max))}')
