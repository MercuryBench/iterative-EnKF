import numpy as np
import scipy.linalg as la
from math import sqrt

""" Global policy: 
Order of dimensions in every numpy array is
-> (optional) time coordinate (artificial time for EnKF iteration)
-> parameter (or observation) space
-> ensemble dimension (or multiple copies)
"""
class Measure:
	# class for prior and observation error measures
	def __init__(self):
		pass
	def sample(self):
		raise NotImplementedError()
		
	
class GaussianMeasure(Measure): # class for a Gaussian measure
	def __init__(self, mu, Sigma): # expexts mu as a 1d numpy array and Sigma as a 2d numpy array
		assert(isinstance(mu, np.ndarray))
		assert(isinstance(Sigma, np.ndarray))
		d = len(mu)
		assert(mu.shape == (d,))
		assert(Sigma.shape == (d,d))
		self.mean = mu
		self.Cov = Sigma
		self.sqrtCov = None
		self.dim = d
	def sample(self, numSamples=1): # samples {numSamples} instances of a random variable distributed according to this measure
		if numSamples == 1:
			return np.random.multivariate_normal(self.mean, self.Cov).reshape(self.dim, 1)
		else:
			return np.random.multivariate_normal(self.mean, self.Cov, numSamples).T
	def KLBasis(self, maxnumelements=-1): # returns the Karhunen Loeve basis (or a subset with {maxnumelements} elements)
		if maxnumelements == -1:
			maxnumelements = self.dim
		b = np.zeros((self.dim, maxnumelements))
		if (self.sqrtCov):
			pass
		else:
			self.sqrtCov = la.sqrtm(self.Cov)
		b = np.dot(self.sqrtCov[:, 0:maxnumelements], np.eye(maxnumelements))
		"""
		for k in range(maxnumelements):
			x = np.zeros((maxnumelements, 1))
			x[0, k] = 1
			b[:, k] = np.dot(self.sqrtCov[:, 0:maxnumelements], x)"""
		return b
class SettingClass:
	# sets up the forward problem with all probabilistic properties:
	# observation y = fwdOp(u) + noise, where u ~ muPrior and noise ~ nuObs
	def __init__(self, muPrior, nuObs, fwdOp):
		assert(isinstance(muPrior, Measure))
		assert(isinstance(nuObs, Measure))
		self.dimPar = muPrior.dim
		self.dimObs = nuObs.dim
		self.prior = muPrior
		self.obsnoise = nuObs
		self.forward = fwdOp # forward needs to be able to handle both single inputs (as a column vector) and multiple inputs (as a matrix consisting
		# of columns containing the inputs
	def samplePrior(self, numSamples=1): # sample (list of) u from prior
		return self.prior.sample(numSamples)
	def sampleObsnoise(self, numSamples=1): # sample (list of) y from noise measure
		return self.obsnoise.sample(numSamples)
	def forward(self, u): # forward dynamics
		assert(isinstance(u, np.ndarray))		
		return self.forward(u)
	def createObservation(self, u):
		assert(isinstance(u, np.ndarray))
		if u.ndim == 2: # in case u consists of multiple inputs
			return self.forward(u) + self.sampleObsnoise(u.shape[1])
		else:
			return self.forward(u) + self.sampleObsnoise()

def EnsembleKalman(T, numIterations, observation, settingClass, J, initialMode="sample", fulloutput=True):
	# implements the Ensemble Kalman method (artificially iterated for numIterations > 1)
	# T is the artifical time
	# numIterations is the number of iterations
	# observation is the data 
	# settingClass contains all information about the forward problem
	# J is the number of ensembles
	# initialMode defines the mode of defining the initial ensemble (by sampling or by Karhunen Loeve expansion)
	# fulloutput gives back all data if True, only means if False
	h = 1.0*T/numIterations
	dimPar = settingClass.dimPar
	dimObs = settingClass.dimObs
	G = settingClass.forward
	if initialMode=="sample":
		u_init = settingClass.samplePrior(J)
	elif initialMode =="KL":
		u_init = settingClass.prior.KLBasis(J)
	else:
		raise NotImplementedError()


	if fulloutput:
		u = np.zeros((numIterations+1, dimPar, J))
		u[0, :, :] = u_init
		perts = settingClass.sampleObsnoise((numIterations, J)) # artificial noise -- will give shape [numIt, J, dimPar], so needs to be swapped
		perts = np.swapaxes(perts, 0, 2)
		perts = np.swapaxes(perts, 1, 2)
		assert(perts.shape == (numIterations, dimObs, J))
		for n in range(numIterations):
			uu = u[n, :, :]
			Guu = G(uu)
			umean = np.mean(uu, axis=1).reshape(dimPar, 1)
			umean_aug = np.tile(umean, (1, J))
			Gumean = np.mean(Guu, axis=1).reshape(dimObs, 1)
			Gumean_aug = np.tile(Gumean, (1, J))
			Cpp = 1.0/J*np.dot(Guu-Gumean_aug, (Guu-Gumean_aug).T)
			Cup = 1.0/J*np.dot(uu-umean_aug, (Guu-Gumean_aug).T)
			assert(Cpp.shape == (dimObs, dimObs))
			assert(Cup.shape == (dimPar, dimObs))
			Gamma = settingClass.obsnoise.Cov
			increment = np.dot(Cup, np.linalg.solve(h*Cpp + Gamma, h*(np.tile(observation, (1, J)) - Guu) + sqrt(h)*perts[n, :, :]))
			u[n+1, :, :] = uu + increment
	return u
	

