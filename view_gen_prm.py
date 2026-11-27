"""Look at individual permutaions of nth gen flower"""

import order_parameter as op
import numpy as np
import sm_in3 as sm
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.optimize import curve_fit
import csv


def get_node_param(g, i, p):

    if i > 2 ** (g - 1):
        raise ValueError("g must be >= 1")
    n = g + 2

    if g == 1:
        TRI = np.array([[0, p**2, p], [p**2, 0, p], [p, p, 0]])
    else:
        TRI = op.repeat_star_to_3(n, i, op.get_sc_array(p, g))

    D_AB, F_TA, E_TB = TRI[0][1], TRI[0][2], TRI[1][2]

    AM, TM, BM = sm.star_3_inv(D_AB, E_TB, F_TA)

    return 0.5 * (TM * AM + TM * BM), AM * BM

P_TH    =   0.5 * (np.sqrt(5) - 1)      ## Analytic critical Value (classical sys.)


def write_to_gen(g,fn, p):
    p_i = []
    for i in range(0,2 ** (g - 2)):
        p_inf, p_sc = get_node_param(g,i,p)
        p_i.append([i,p_inf, p_sc])
        #print(f'n={g}, i={i}, p={p}, p_inf={p_inf} p_sc={p_sc}')

    
    p_i = np.array(p_i)

    #plt.scatter(p_i.T[0],p_i.T[1],s=4)    ################################ <----------------
    #plt.show()
    with open(f'{fn}.csv', 'w',newline='') as file:
        write = csv.writer(file)
        write.writerows(p_i)


def read_to_gen(fn, g,p, nobins = 20):
    t = np.loadtxt(f'{fn}.csv',delimiter=',').T
    plt.hist(t[1],bins=nobins,label=f'n={p}')
    plt.hist(t[2],bins=nobins,label=f'n={p}')
    #print(t[0])
    #plt.savefig(f'figures/{fn}.png',dpi=500)
    #plt.show()

if __name__ == '__main__':

    G=17

    write_to_gen(G,'ttttr', P_TH)
    #write_to_gen(13,'n12p0.1', P_TH-0.01)
    #write_to_gen(13,'n12p0.0', P_TH)
    #write_to_gen(13,'n12p0.6', P_TH+0.01)
    #write_to_gen(13,'n12p0.2', P_TH+0.05)
    #write_to_gen(13,'n12p0.9', P_TH+0.2)

    #read_to_gen('tst',G, P_TH, 50)
    #read_to_gen('n12p0.1',13, P_TH-0.01, 100)
    #read_to_gen('n12p0.0',13, P_TH, 50)
    #read_to_gen('n12p0.6',13, P_TH+0.01, 30)
    #read_to_gen('n18',18, P_TH, 300)
#
    plt.legend()
    plt.savefig('figures/allinall.png',dpi=500)
    plt.show()


def the_rest():
    pass
    #counts, edges = np.histogram(p_i, bins=200, density=True)
#
    #centres = 0.5 * (edges[1:] + edges[:-1])
    #
    #dc = edges[1]-edges[0]
    #
    #area = np.sum(dc*counts)
#
    #def func(x,a,b):
    #    return a*(x**2)*np.exp(-(b*x)**2)
#
    #H = curve_fit(func, centres, counts, [20000,27])
#
    #plt.xlim(0.,.5)
    #plt.scatter(centres,counts)
    #plt.plot(np.linspace(0,0.4,100),func(np.linspace(0,0.4,100),H[0][0],H[0][1]))
    #plt.show()

    #print(f"Poisson fit: lambda = {lam:.4f} +- {lam_err:.4f}")
    #print(f"sample mean = {x.mean():.4f}, sample variance = {x.var():.4f}