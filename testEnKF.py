import numpy as np
from EnsembleKalman import *
from math import pi, sin, cos, sqrt
import matplotlib.pyplot as plt

# setting params
NModes = 200	# number of modes in spectral decomposition 

N = NModes
hSpace = 0.001
x = np.arange(0.0, 2*pi, hSpace)
THeat = 0.001;

modesfnc = 1/sqrt(pi)*np.concatenate((np.cos(np.dot(np.arange(1, N/2+1).reshape(N/2, 1), x.reshape(1, len(x)) )), np.sin(np.dot( np.arange(1, N/2+1).reshape(N/2, 1), x.reshape(1, len(x)) ))), axis=0)

# prior params
beta = .5;		
alpha = .5 + .2;


mu = np.array([0])
mu = np.concatenate((mu, -2/(pi*(np.arange(2, N/2+1)**2-1))*(1-np.mod(np.arange(2, N/2+1), 2))), axis=0)
mu = np.concatenate((mu, np.array([0.5])), axis=0)
mu = np.concatenate((mu, np.zeros(N/2-1)), axis=0)

Sigma = beta**2*np.diag(np.array(np.concatenate((np.arange(1.0, N/2+1)**(-2*alpha), np.arange(1.0, N/2+1)**(-2*alpha)), axis=0)))

prior = GaussianMeasure(mu, Sigma)


# obs and obs noise params

NObs = 30
gamma = .2;		

nu = np.zeros(NObs)

Gamma = gamma**2*np.eye(NObs)

obsnoise = GaussianMeasure(nu, Gamma)

# forward operator

def forwardHeatEq(u_ms, THeat):
	if u_ms.ndim > 1:
		(N, J) = u_ms.shape
		factors = np.concatenate((np.array([np.arange(1.0, N/2+1)**2]).T, np.array([np.arange(1.0, N/2+1)**2]).T), axis=0)
		multcoeffs = np.tile( np.exp(-factors*THeat ), (1, J) ) 
	else:
		N = len(u_ms)
		u_ms.reshape(N, 1)
		factors = np.concatenate((np.array([np.arange(1.0, N/2+1)**2]).T, np.array([np.arange(1.0, N/2+1)**2]).T), axis=0)
		multcoeffs = np.exp(-factors)	
	return multcoeffs*u_ms

def obsOp_param(v_ms, NObs, mfnc):
	indicesObs = findIndices(x, np.linspace(min(x), max(x), NObs))
	v = np.dot(mfnc.T, v_ms)	# from fourier space into real space
	if v_ms.ndim > 1:
		(N, J) = v_ms.shape		
		return v[indicesObs, :]
	else:
		N = len(v_ms)
		return v[indicesObs]
		
	

def findIndices(xlong, xshort):
	# for every x in xshort, finds index of value x' in xlong being closest to x
	# assumes xlong and xshort is ordered
	indices = np.zeros(xshort.shape)
	for m, x in enumerate(xshort):
		for n, xp in enumerate(xlong):
			if xp > x:
				if n > 0:
					if abs(xlong[n-1]-x) < abs(xlong[n]-x):
						indices[m] = n-1
					else:
						indices[m] = n
				else:
					indices[m] = 0
				break
			if n == len(xlong)-1: # last possible element
				indices[m] = n
	return indices.astype(int)

def fwd(u_ms):
	return forwardHeatEq(u_ms, THeat)

def obs(v_ms):
	return obsOp_param(v_ms, NObs, modesfnc)

utruth_ms = prior.sample()
v_ms = fwd(utruth_ms)
gv = obs(v_ms)
y = gv + obsnoise.sample()
indicesObs = findIndices(x, np.linspace(min(x), max(x), NObs))
xObs = x[indicesObs]


def G(u_ms):
	return obs(fwd(u_ms))

sc = SettingClass(prior, obsnoise, G)


T = 1
numIterations = 1
observation = y
settingClass = sc
J = 200


res = EnsembleKalman(T, numIterations, y, sc, J, initialMode="sample", fulloutput=True) # use also option initialMode = "KL" for Karhunen-Loeve
resmean = np.mean(res[-1, :, :], axis=1).reshape(N, 1)
plt.subplot(2, 1, 1)
plt.plot(x, np.dot(modesfnc.T, utruth_ms), 'g')

plt.plot(xObs, y, 'ro')
#plt.show()
plt.plot(x, np.dot(modesfnc.T, resmean), 'b')
plt.subplot(2, 1, 2)
plt.plot(x, np.dot(modesfnc.T, v_ms), 'g')
plt.plot(x, np.dot(modesfnc.T, fwd(resmean)), 'b')
plt.show()

