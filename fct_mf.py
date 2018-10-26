import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.optimize import fsolve
from scipy.integrate import quad
from functools import partial

import fct_facilities as fac
from fct_integrals import *

#### Global variables for Gaussian quadrature

gaussian_norm = (1/np.sqrt(np.pi))
gauss_points,gauss_weights = np.polynomial.hermite.hermgauss(200)
gauss_points = gauss_points*np.sqrt(2)

### Solve the DMF equation through ITERATION

def OpenLoopStatic (y0, g, VecPar, tolerance=1e-12, backwards = 1):  

    again = 1
    y = np.array(y0)
    y_new = np.ones(2)
    eps = 0.2

    A, Sim, Sii, rho = VecPar
    Simi = Sim * Sii * rho**2

    while (again==1):

        # Take a step

        new0 = A
        new1 = g*g * PhiSq(0, y[1]) + (Sim*y[0])**2 + 2*Simi*y[0] + Sii**2

        y_new[0] = (1-backwards*eps)*y[0] + eps*backwards*new0
        y_new[1] = (1-eps)*y[1] + eps*new1

        # Stop if the variance converge to a number

        if( (np.fabs(y[0]-y_new[0])<tolerance*np.fabs(y[0])) and (np.fabs(y[1]-y_new[1])<tolerance*np.fabs(y[1])) ):
            again=0

        if( np.fabs(y[0]-y_new[0])<tolerance ):
            again=0

        if np.isnan(y_new[0]) == True:
            print 'Nan'
            again = 0
            y_new = [0,0]

        if( np.fabs(y[0])> 1/tolerance  ):
            print 'Explosion'
            again=0
            y_new = [0,0]
    
        y[0]=y_new[0]
        y[1]=y_new[1]

    return [y_new[0], np.fabs(y_new[1])]


def ClosedLoopStatic (y0, g, VecPar, p, pm, pI, tolerance=1e-12, backwards = 1):  

    again = 1
    y = np.array(y0)
    y_new = np.ones(2)
    eps = 0.2

    A, Sim, Sii, rho = VecPar
    Simi = Sim * Sii * rho**2

    while (again==1):

        # Take a step

        new0 = ( (p*rho*Sim + pm*np.sqrt(1-rho**2)*Sim)* y[0] + (p*rho*Sii + pI*np.sqrt(1-rho**2)*Sii) ) * Prime(0, y[1])
        new1 = g*g * PhiSq(0, y[1]) + (Sim*y[0])**2 + 2*Simi*y[0] + Sii**2

        y_new[0] = (1-backwards*eps)*y[0] + eps*backwards*new0
        y_new[1] = (1-eps)*y[1] + eps*new1

        # Stop if the variance converge to a number

        if( (np.fabs(y[0]-y_new[0])<tolerance*np.fabs(y[0])) and (np.fabs(y[1]-y_new[1])<tolerance*np.fabs(y[1])) ):
            again=0

        if( np.fabs(y[0]-y_new[0])<tolerance ):
            again=0

        if np.isnan(y_new[0]) == True:
            print 'Nan'
            again = 0
            y_new = [0,0]

        if( np.fabs(y[0])> 1/tolerance  ):
            print 'Explosion'
            again=0
            y_new = [0,0]
    
        y[0]=y_new[0]
        y[1]=y_new[1]

    return [y_new[0], np.fabs(y_new[1])]


def EigStationary (g, VecPar, sol, p, pm, pI):

    A, Sim, Sii, rho = VecPar
    Simi = Sim * Sii * rho**2

    stability_matrix = np.zeros(( 3, 3 ))

    stability_matrix[0,0] = 0
    stability_matrix[0,1] = 0
    stability_matrix[0,2] = 0

    stability_matrix[1,0] = 2 * ( g**2*PhiPrime(0, sol[1]))
    stability_matrix[1,1] = g**2*(PrimeSq(0, sol[1])+ PhiSec (0, sol[1])) 
    stability_matrix[1,2] = 2*Sim**2*sol[0] + 2*Simi

    a = (p * rho* Sim + pm * Sim * np.sqrt(1-rho**2)) * Prime(0, sol[1])
    b = 0.5 * ( (p*rho*Sim + pm*np.sqrt(1-rho**2)*Sim)* sol[0] + (p*rho*Sii + pI*np.sqrt(1-rho**2)*Sii) ) * Third(0, sol[1])

    stability_matrix[2,0] = b*stability_matrix[1,0]
    stability_matrix[2,1] = b*stability_matrix[1,1]
    stability_matrix[2,2] = b*stability_matrix[1,2] + a

    eig = np.linalg.eigvals(stability_matrix)

    eig = eig [np.fabs(eig.real)>1e-15] # Cut the zero eigenvalue

    return np.flipud(np.sort(eig)) 

