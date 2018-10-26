#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

#### Supplementary code for the paper: 
#### "Characterizing global stability in trained recurrent networks"
#### F. Mastrogiuseppe and S. Ostojic (2018)

#### This code solves the mean-field equations for a readout in the form of Eq. 9,
#### for arbitrary values of p, pm and pI
#### Details as in Fig. 1e

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

### Import functions

import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append('')

import fct_facilities as fac
import fct_integrals as integ
import fct_mf as mf


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Fix the parameters

g = 0.3

# Input vectors parameters

Sim = 1.2 # variance of m
Sii = 0.5 # variance of I
rho = 0.5 # correlation coefficient (between 0 to 1)

# Readout parameters (see Eq. 9)

_p = 1.
_pm = 1.
_pI = 0.3
flex = - (Sii * rho * _p + Sii * np.sqrt(1-rho**2) * _pI) / ( Sim * rho * _p + Sim * np.sqrt(1-rho**2) * _pm )


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Compute mean-field solutions

doCompute = 1

A_values = np.linspace(-2, 2, 1e3)

readout = np.zeros(( 3, len(A_values) ))
outlier = np.zeros(( 3, len(A_values) ))
radius = np.zeros(( 3, len(A_values) ))

if doCompute:

	# Initial conditions for iterating the mean-field equations

	ic_p = [5., 10.]
	ic_m = [-2, 10.]

	for i, A in enumerate(A_values):

		if ( A< -0.26 or A>-0.2 ): # Exclude narrow region where iteration in this form is unstable

			print A

			ParVec = [A, Sim, Sii, rho]


			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
			# Open loop

			z_ol, delta0_ol = mf.OpenLoopStatic (ic_p, g, ParVec)

			factor = Sii * rho * _p + Sii * np.sqrt(1-rho**2) * _pI + A * ( Sim * rho * _p + Sim * np.sqrt(1-rho**2) * _pm )
			C = A / ( factor * integ.Prime(0, delta0_ol) )

			# Normalized weigths
			p = C * _p
			pm = C * _pm
			pI = C * _pI


			#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
			# Closed loop

			readout[0,i], delta0_cl = mf.ClosedLoopStatic (ic_p, g, ParVec, p, pm, pI)

			eig = mf.EigStationary (g, ParVec, [readout[0,i], delta0_cl], p, pm, pI)
			outlier[0,i] = eig[0]
			radius[0,i] = g * np.sqrt ( integ.PrimeSq(0, delta0_cl) )
			ic_p = [ readout[0,i], delta0_cl ]

			#

			readout[1,i], delta0_cl = mf.ClosedLoopStatic (ic_m, g, ParVec, p, pm, pI)

			eig = mf.EigStationary (g, ParVec, [readout[1,i], delta0_cl], p, pm, pI)
			outlier[1,i] = eig[0]
			radius[1,i] = g * np.sqrt ( integ.PrimeSq(0, delta0_cl) )
			if A<0: ic_m = [ - np.fabs( readout[1,i]) , delta0_cl ]
			else: ic_m = [-2., 0.1]
			
			#

			readout[2,i], delta0_cl = mf.ClosedLoopStatic ([0.8*readout[1,i], delta0_cl], g, ParVec, p, pm, pI, backwards = -1)

			eig = mf.EigStationary (g, ParVec, [readout[2,i], delta0_cl], p, pm, pI) 
			outlier[2,i] = eig[0]
			radius[2,i] = g * np.sqrt ( integ.PrimeSq(0, delta0_cl) )

	# Store

	fac.Store(readout, 'readout.p', 'VaryA/')
	fac.Store(outlier, 'outlier.p', 'VaryA/')
	fac.Store(radius, 'radius.p', 'VaryA/')

else:

	# Retrieve

	readout = fac.Retrieve ('readout.p', 'VaryA/')
	outlier = fac.Retrieve ('outlier.p', 'VaryA/')
	radius = fac.Retrieve ('radius.p', 'VaryA/')


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Plot

fac.SetPlotParams()
dashes = [3,3]

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Readout 

fg = plt.figure()
ax0 = plt.axes(frameon=True)

plt.plot (A_values, A_values, linewidth = 4, color = '#95B3DF')

condition_p = np.logical_and(A_values < flex, readout[0,:]>0)
condition_n = np.logical_and(A_values < flex, readout[1,:]<0)

plt.plot(A_values[condition_p], readout[0,:][condition_p], color = '#21406F')
plt.plot(A_values[condition_n], readout[1,:][condition_n], color = '#21406F')
line, = plt.plot(A_values[condition_n], readout[2,:][condition_n], color = '#21406F')
line.set_dashes(dashes)

condition_p = np.logical_and(A_values > flex, np.fabs(readout[0,:])>0)
condition_n = np.logical_and(A_values > 0, readout[1,:]<0)

plt.plot(A_values[condition_p], readout[0,:][condition_p], color = '#21406F')
plt.plot(A_values[condition_n], readout[1,:][condition_n], color = '#21406F')
line, = plt.plot(A_values[condition_n], readout[2,:][condition_n], color = '#21406F')
line.set_dashes(dashes)

plt.xlabel(r'Target value')
plt.ylabel('Readout $z$')

plt.xlim(-2, 2)
plt.ylim(-4, 4)

plt.legend(loc=3)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.savefig('readout_A.pdf')
plt.show()


#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 
### Outlier 

fg = plt.figure()
ax0 = plt.axes(frameon=True)

# Radius

condition_p = np.logical_and(A_values < flex, readout[0,:]>0)
condition_n = np.logical_and(A_values < flex, readout[1,:]<0)

plt.plot(A_values[condition_p], radius[0,:][condition_p], color = '0.7')
plt.plot(A_values[condition_n], radius[1,:][condition_n], color = '0.7')
line, = plt.plot(A_values[condition_n], radius[2,:][condition_n], color = '0.7')
line.set_dashes(dashes)

condition_p = np.logical_and(A_values > flex, np.fabs(readout[0,:])>0)
condition_n = np.logical_and(A_values > 0, readout[1,:]<0)

plt.plot(A_values[condition_p], radius[0,:][condition_p], color = '0.7')
plt.plot(A_values[condition_n], radius[1,:][condition_n], color = '0.7')
line, = plt.plot(A_values[condition_n], radius[2,:][condition_n], color = '0.7')
line.set_dashes(dashes)

# Outlier 

condition_p = np.logical_and(A_values < flex, readout[0,:]>0)
condition_n = np.logical_and(A_values < flex, readout[1,:]<0)

plt.plot(A_values[np.logical_and(condition_p, outlier[0,:]<1)], outlier[0,:][np.logical_and(condition_p, outlier[0,:]<1)], color = '#006666')
plt.plot(A_values[np.logical_and(condition_n, outlier[1,:]<1)], outlier[1,:][np.logical_and(condition_n, outlier[1,:]<1)], color = '#006666')
line, = plt.plot(A_values[condition_n], outlier[2,:][condition_n], color = '#006666')
line.set_dashes(dashes)

condition_p = np.logical_and(A_values > flex, np.fabs(readout[0,:])>0)
condition_n = np.logical_and(A_values > 0, readout[1,:]<0)

plt.plot(A_values[condition_p], outlier[0,:][condition_p], color = '#006666')
plt.plot(A_values[condition_n], outlier[1,:][condition_n], color = '#006666')
line, = plt.plot(A_values[condition_n], outlier[2,:][condition_n], color = '#006666')
line.set_dashes(dashes)

plt.xlabel(r'Target value')
plt.ylabel('Outlier eigenvalue')

plt.legend(loc=3)

plt.xlim(-2, 2)
plt.ylim(0, 3)

ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.yaxis.set_ticks_position('left')
ax0.xaxis.set_ticks_position('bottom')
plt.locator_params(nbins=4)

plt.savefig('outlier_A.pdf')
plt.show()

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### 

sys.exit(0)
