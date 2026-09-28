""" perc_models.py """

import numpy as np


def R_P(p):
    return 1 - (1 - p**2)**2

def P_SC(p, n):
    for _ in range(n):
        p = R_P(p)
    return p

def g(c):
    return (1 + np.sqrt(np.clip(1 - c**2, 0, None))) / 2

def parallel_c(c):
    g_p = np.maximum(0.5, g(c) ** 2)
    return np.sqrt(np.maximum(0.0, 1 - (2*g_p - 1)**2))

def R_C(c):
    branch = c**2   ## series rule
    return parallel_c(branch)   ## parallel rule

def C_SC(c, n):
    for _ in range(n):
        c = R_C(c)
    return c
