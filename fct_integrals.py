import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import fsolve
from scipy.integrate import quad
from functools import partial

import fct_facilities as fac

#### Global variables for Gaussian quadrature

gaussian_norm=(1/np.sqrt(np.pi))
gauss_points,gauss_weights=np.polynomial.hermite.hermgauss(200)
gauss_points=gauss_points*np.sqrt(2)

#### Gaussian integrals with QUADRATURE

#

def Prim (mu, delta0):
    integrand=np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points))
    return gaussian_norm * np.dot (integrand,gauss_weights)

def Phi (mu, delta0):
    integrand=np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def Prime (mu, delta0):
    integrand=1-(np.tanh(mu+np.sqrt(delta0)*gauss_points))**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def Sec (mu, delta0):
    integrand = - 2 * np.tanh(mu+np.sqrt(delta0)*gauss_points) * (1-(np.tanh(mu+np.sqrt(delta0)*gauss_points))**2)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def Third (mu, delta0):
    integrand = - 2 * ( - 3*np.tanh(mu+np.sqrt(delta0)*gauss_points)**2 + 1) * (1-(np.tanh(mu+np.sqrt(delta0)*gauss_points))**2)
    return gaussian_norm * np.dot (integrand,gauss_weights)

#

def PrimSq (mu, delta0):
    integrand=np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points))
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

def PhiSq (mu, delta0):
    integrand=np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

def PrimeSq (mu, delta0):
    integrand=1-(np.tanh(mu+np.sqrt(delta0)*gauss_points))**2
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

#

def PhiPrime (mu, delta0):
    integrand=np.tanh(mu+np.sqrt(delta0)*gauss_points) * (1-(np.tanh(mu+np.sqrt(delta0)*gauss_points))**2)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PrimPrime (mu, delta0):
    integrand=(1-(np.tanh(mu+np.sqrt(delta0)*gauss_points))**2) * np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points))
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PhiSec (mu, delta0):
    integrand= - 2 * (np.tanh(mu+np.sqrt(delta0)*gauss_points)**2 ) * (1-(np.tanh(mu+np.sqrt(delta0)*gauss_points))**2)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def PrimPhi (mu, delta0):
    integrand=np.log(np.cosh(mu+np.sqrt(delta0)*gauss_points)) * np.tanh(mu+np.sqrt(delta0)*gauss_points)
    return gaussian_norm * np.dot (integrand,gauss_weights)

#

def InnerPrimPrim (z, mu, delta0, deltainf):
    integrand=np.log(np.cosh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z))
    return gaussian_norm * np.dot (integrand,gauss_weights)

def IntPrimPrim (mu, delta0, deltainf): # Performs the integral over z
    integrand=InnerPrimPrim(gauss_points, mu, delta0, deltainf)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

InnerPrimPrim=np.vectorize(InnerPrimPrim)


def InnerPhiPhi (z, mu, delta0, deltainf):
    integrand=np.tanh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def IntPhiPhi (mu, delta0, deltainf): # Performs the integral over z
    integrand=InnerPhiPhi(gauss_points, mu, delta0, deltainf)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

InnerPhiPhi=np.vectorize(InnerPhiPhi)


def InnerPrimePrime (z, mu, delta0, deltainf):
    integrand=1-np.tanh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z)**2
    return gaussian_norm * np.dot (integrand,gauss_weights)

def IntPrimePrime (mu, delta0, deltainf): # Performs the integral over z
    integrand=InnerPrimePrime(gauss_points, mu, delta0, deltainf)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

InnerPrimePrime=np.vectorize(InnerPrimePrime)

#

def InnerPrimPhi (z, mu, delta0, deltainf):

    integrand=np.log(np.cosh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z))
    int1 = gaussian_norm * np.dot (integrand,gauss_weights)

    integrand=np.tanh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z)
    int2 = gaussian_norm * np.dot (integrand,gauss_weights)

    return int1 * int2

def IntPrimPhi (mu, delta0, deltainf): # Performs the integral over z
    integrand=InnerPrimPhi(gauss_points, mu, delta0, deltainf)
    return gaussian_norm * np.dot (integrand,gauss_weights)

InnerPrimPhi=np.vectorize(InnerPrimPhi)


def InnerPrimPrime (z, mu, delta0, deltainf):

    integrand=np.log(np.cosh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z))
    int1 = gaussian_norm * np.dot (integrand,gauss_weights)

    integrand=1-np.tanh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z)**2
    int2 = gaussian_norm * np.dot (integrand,gauss_weights)

    return int1 * int2

def IntPrimPrime (mu, delta0, deltainf): # Performs the integral over z
    integrand=InnerPrimPrime(gauss_points, mu, delta0, deltainf)
    return gaussian_norm * np.dot (integrand,gauss_weights)

InnerPrimPrime=np.vectorize(InnerPrimPrime)


def InnerPhiPrime (z, mu, delta0, deltainf):

    integrand=np.tanh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z)
    int1 = gaussian_norm * np.dot (integrand,gauss_weights)

    integrand=1-np.tanh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z)**2
    int2 = gaussian_norm * np.dot (integrand,gauss_weights)

    return int1 * int2

def IntPhiPrime (mu, delta0, deltainf): # Performs the integral over z
    integrand=InnerPhiPrime(gauss_points, mu, delta0, deltainf)
    return gaussian_norm * np.dot (integrand,gauss_weights)

InnerPhiPrime=np.vectorize(InnerPhiPrime)


def InnerPhiSec (z, mu, delta0, deltainf):

    integrand=np.tanh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z)
    int1 = gaussian_norm * np.dot (integrand,gauss_weights)

    integrand = - 2 * np.tanh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z) * (1-(np.tanh(mu+np.sqrt(delta0-deltainf)*gauss_points+np.sqrt(deltainf)*z))**2)
    int2 = gaussian_norm * np.dot (integrand,gauss_weights)

    return int1 * int2

def IntPhiSec (mu, delta0, deltainf): # Performs the integral over z
    integrand=InnerPhiSec(gauss_points, mu, delta0, deltainf)
    return gaussian_norm * np.dot (integrand,gauss_weights)

InnerPhiSec=np.vectorize(InnerPhiSec)


#


def InnerPhiPattern (z, mu, var_in, var_out):
    integrand=np.tanh(mu+np.sqrt(var_in)*gauss_points+np.sqrt(var_out)*z)
    return gaussian_norm * np.dot (integrand,gauss_weights)

def IntPhiPattern (mu, var_in, var_out): # Performs the integral over z
    integrand=InnerPhiPattern(gauss_points, mu, var_in, var_out)
    return gaussian_norm * np.dot (integrand**2,gauss_weights)

InnerPhiPattern=np.vectorize(InnerPhiPattern)

