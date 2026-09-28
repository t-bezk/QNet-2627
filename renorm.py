import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

H_OUT = []
H_OUT2 = []
P_STR = 0.5 * (np.sqrt(5) - 1)

M0 = 3
M1 = 16
M2 = 30
P_LO = 0.4
P_HI = 0.8

def R_P(p):
    return 1 - (1 - p**2)**2

def P_SC(p, n):
    for _ in range(n):
        p = R_P(p)
    return p

def g(c):
    return (1 + np.sqrt(np.clip(1 - c**2, 0, None))) / 2

def parallel_c(c):
    g_p = max(0.5, g(c) ** 2)
    return np.sqrt(max(0.0, 1 - (2*g_p - 1)**2))

def R_C(c):
    branch = c**2          # series rule: two links in a row
    return parallel_c(branch)   # parallel rule: two branches

def C_SC(c, n):
    for _ in range(n):
        c = R_C(c)
    return c


def parameter_scaling(prm_th, x_sc, x_str, display):
    """
    Order parameter scaling for X-type percolation; returns nu as float.

    Args:
        prm_th (float) - parameter value
        x_sc (function) - 
        x_inf (float) - critical threshold for infinite clusters
    """
    prm_n = []
    tht_q = []

    for _n in range(M0,M2):
        ## determine threshold probability for given generation
        x_th_l = op.brentq(lambda x: x_sc(x, _n) - prm_th, 1e-8, 1 - 1e-8)

        if _n >= M1-1:
            continue

        prm_n.append(x_th_l)
        tht_q.append(2**_n)

    plt.savefig('P_inf.png',dpi=400)
    plt.close()

    diff_tm = np.asarray(x_th_l) if x_str is None else np.asarray(x_str)
    log_plot = np.log10(np.abs(prm_n - diff_tm))
    grad_fit = np.polyfit(np.log10(tht_q), log_plot, 1)

    plt.scatter(np.log10(tht_q),log_plot)
    plt.plot(np.log10(tht_q), grad_fit[0]*np.log10(tht_q) + grad_fit[1])

    if not display:
        plt.close()

    ## nu recovered from inverse gradient
    return -1/grad_fit[0]


H_OUT.append(parameter_scaling(P_LO, P_SC, P_STR, 1))
H_OUT.append(parameter_scaling(P_HI, P_SC, P_STR, 1))
plt.show()
print(f'{0.5*(H_OUT[0] + H_OUT[1])} pm {0.5*(H_OUT[0] - H_OUT[1])}')

H_OUT2.append(parameter_scaling(P_LO, C_SC, None, 1))
H_OUT2.append(parameter_scaling(P_HI, C_SC, None, 1))
plt.show()
print(f'{0.5*(H_OUT2[0] + H_OUT2[1])} pm {0.5*(H_OUT[0] - H_OUT[1])}')
