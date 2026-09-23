import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op


def R_P(p):
    return 1 - (1 - p**2)**2

def P_SC(p, n):
    for _ in range(n):
        p = R_P(p)
    return p

def g(c):
    return (1 + np.sqrt(np.clip(1 - c**2, 0, None))) / 2

def parallel_c(c):
    gp = max(0.5, g(c) ** 2)
    return np.sqrt(max(0.0, 1 - (2*gp - 1)**2))

def R_C(c):
    branch = c**2          # series rule: two links in a row
    return parallel_c(branch)   # parallel rule: two branches

def C_SC(c, n):
    for _ in range(n):
        c = R_C(c)
    return c


theta = np.linspace(0, np.pi/4, 10000)
p = 2 * np.sin(theta)**2

L = []
E = []

M0 = 4
M = 50
P_TH = [0.4,0.8]

H_OUT = []

for OP in P_TH:
    P_STR = 0.5 * (np.sqrt(5) - 1)

    for n in range(M0,M):
        ## determine threshold probability for given generation
        p_th_L = op.brentq(lambda p: C_SC(p, n) - OP, 1e-8, 1 - 1e-8)

        if n >= M-1:
            continue
        print(p_th_L)

        E.append(p_th_L)
        L.append(2**n)

    plt.savefig('P_inf.png',dpi=400)
    plt.close()

    logplot = np.log10(np.abs(E - np.asarray(p_th_L)))

    plt.scatter(np.log10(L),logplot)

    H = np.polyfit(np.log10(L), logplot, 1)

    plt.plot(np.log10(L), H[0]*np.log10(L) + H[1])

    H_OUT.append(-1/H[0])

print(f'{H_OUT}')


plt.show()
