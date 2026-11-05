import numpy as np

from scipy.optimize import brentq

from sympy import *



def p_3(a,b,c):
    return a*b - a*c - b*c + 1

def p_2(a,b,c):
    return -a*a*b*b + a*a*b*c + a*b*b*c - a*b*c*c - a*b + a*c + b*c - 1

def p_1(a,b,c):
    return a*b*c*c + b*b*c*c - a*a*b*c - a*b*b*c + a*b - 1

def p_0(a,b,c):
    return a*b * (1-c*c)

def f_cbic(f,a,b,c):
    return p_3(a,b,c)*f**3 + p_2(a,b,c)*f**2 + p_1(a,b,c)*f + p_0(a,b,c)

def d_evte(a,b,c,f):
    return (a*c + (a*b-b*c)*f-f**2)/(1-f**2)

def e_evte(a,b,c,f):
    return (b*c + (a*b-a*c)*f-f**2)/(1-f**2)

def star_3_fwd(a,b,c):
    """ Returns a fast numerical approximation for the forward star-mesh transformation"""

    f_0 = brentq(lambda x: f_cbic(x,a,b,c), 0, 1, xtol=1e-13)
    d_0 = d_evte(a,b,c,f_0)
    e_0 = e_evte(a,b,c,f_0)

    return d_0, e_0, f_0


d, e, f = star_3_fwd(0.6,0.2,0.8)

print(f'{d} {e} {f}')