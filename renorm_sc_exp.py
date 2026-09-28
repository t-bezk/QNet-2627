""" renorm_sc_exp.py """

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op
from perc_models import P_SC, C_SC

H_OUT = []
H_OUT2 = []
P_STR = 0.5 * (np.sqrt(5) - 1)
DP = True   ## display

M0 = 1  ## starting generation index
M1 = 13 ## max generation index
M2 = 45 ## generation of approx threshold (should be large)
P_LO = 0.4
P_HI = 0.8


theta = np.linspace(0, np.pi/4, 1000)

def parameter_scaling(prm_th, x_sc, x_str, display,t=''):
    """
    Order parameter scaling for X-type percolation;
        Returns critical exponent nu as float.

    Args:
        prm_th (float) - parameter value
        x_sc (function) - system model
        x_inf (float) - critical threshold for infinite clusters
    """
    prm_n = []
    tht_q = []

    plt.close()

    for _n in range(M0,M2):
        ## determine threshold probability for given generation
        x_th_l = op.brentq(lambda x: x_sc(x, _n) - prm_th, 1e-8, 1 - 1e-8)

        if _n >= M1-1 and _n < M2:
            continue

        prm_n.append(x_th_l)
        tht_q.append(2**_n)

        if display:
            plt.plot(theta, x_sc(theta, _n))

    plt.savefig(f'{t}_inf_atn.png',dpi=400)
    plt.close()

    ## log of difference as prm_n approaches threshold
    diff_tm = np.asarray(x_th_l) if x_str is None else np.asarray(x_str)
    log_plot = np.log10(np.abs(prm_n - diff_tm))
    grad_fit = np.polyfit(np.log10(tht_q), log_plot, 1)

    plt.scatter(np.log10(tht_q),log_plot)
    plt.plot(np.log10(tht_q), grad_fit[0]*np.log10(tht_q) + grad_fit[1])

    ## nu recovered from inverse gradient
    return -1/grad_fit[0]


p_nu_min = parameter_scaling(P_LO, P_SC, P_STR, DP,t='P')
p_nu_max = parameter_scaling(P_HI, P_SC, P_STR, DP,t='P')
plt.show()
print(f'v = {0.5*(p_nu_min + p_nu_max)} ± {abs(0.5*(p_nu_min - p_nu_max))}')

c_nu_min = parameter_scaling(P_LO, C_SC, None, DP,t='C')
c_nu_max = parameter_scaling(P_HI, C_SC, None, DP,t='C')
plt.show()
print(f'v = {0.5*(c_nu_min + c_nu_max)} ± {abs(0.5*(c_nu_min - c_nu_max))}')
