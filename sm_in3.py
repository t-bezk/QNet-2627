import numpy as np

from scipy.optimize import brentq




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

def d_evte(a,b,c,f):## Implement divide by zero check
    return (a*c + (a*b-b*c)*f-f**2)/(1-f**2)

def e_evte(a,b,c,f):## Implement divide by zero check
    return (b*c + (a*b-a*c)*f-f**2)/(1-f**2)

def a_evte_inv(d,e,f): ## Implement divide by zero check
    return ((d*e+f-d*e*f)*(e*f+d-d*e*f)/(d*f+e-d*e*f))**0.5

def b_evte_inv(d,e,f):## Implement divide by zero check
    return ((d*e+f-d*e*f)*(d*f+e-d*e*f)/(e*f+d-d*e*f))**0.5

def c_evte_inv(d,e,f):## Implement divide by zero check
    return ((e*f+d-d*e*f)*(d*f+e-d*e*f)/(d*e+f-d*e*f))**0.5


def star_3_fwd(a,b,c):
    """ Returns a fast numerical approximation for the forward star-mesh transformation"""

    f_0 = brentq(lambda x: f_cbic(x,a,b,c), 0, 1, xtol=1e-13)
    d_0 = d_evte(a,b,c,f_0)
    e_0 = e_evte(a,b,c,f_0)

    return d_0, e_0, f_0

def star_3_inv(d,e,f):
    """ Returns exact inverse star-mesh for a 3-edge system """

    a_0 = a_evte_inv(d,e,f)
    b_0 = b_evte_inv(d,e,f)
    c_0 = c_evte_inv(d,e,f)

    return a_0, b_0, c_0
