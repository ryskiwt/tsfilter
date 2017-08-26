import numpy as np
from numpy import linalg, random
from scipy import special
from numba import jit, i2, f8


class Filter(object):
    def __init__(self, update, loglikelihood, estimate, dtype=np.float64, ess_threshold_ratio=0.9):
        self.dtype = dtype
        self.update = update
        self._loglikelihood = loglikelihood
        self.estimate = estimate
        self.ess_threshold_ratio = ess_threshold_ratio

        self.pars = None
        self.weights = None


    def initialize(self, pars0):
        self.num = pars0.shape[1]
        self.pars = pars0.astype(self.dtype).copy()
        self.weights = np.ones(self.num).astype(self.dtype)
        self.weights[:] = self.weights / np.sum(self.weights)


    def loglikelihood(self, s, hook=None):
        threshold = self.num * self.ess_threshold_ratio
        pars = self.pars
        weights = self.weights
        num = self.num

        n = s.shape[1]
        A = np.empty(n).astype(self.dtype)

        def h(pf, y): pass
        hook = hook if hook is not None else h

        for i in range(n):
            y = s[:,i].reshape(-1,1)
            hook(self, y)

            self.update(pars)
            weights[:] = np.exp( np.log(weights) + self._loglikelihood(y, pars) )
            weights[:] = weights / np.sum(weights)

            A[i] = np.sum(weights)

            ess = 1 / np.sum(weights**2)
            if ess < threshold:
                idx = resample(num, np.cumsum(weights))
                pars[:] = pars[:,idx]
                weights[:] = weights[idx]
                weights[:] = weights / np.sum(weights)

        return np.sum(np.log(A)) - n * np.log(num)


    def filtering(self, s, hook=None):
        threshold = self.num * self.ess_threshold_ratio
        pars = self.pars
        weights = self.weights
        num = self.num

        n = s.shape[1]
        s2 = np.empty(s.shape).astype(self.dtype)

        def h(pf, y): pass
        hook = hook if hook is not None else h

        for i in range(n):
            y = s[:,i].reshape(-1,1)
            hook(self, y)

            self.update(pars)
            weights[:] = np.exp( np.log(weights) + self._loglikelihood(y, pars) )
            weights[:] = weights / np.sum(weights)

            s2[:,i] = self.estimate(pars, weights).ravel()

            ess = 1 / np.sum(weights**2)
            if ess < threshold:
                idx = resample(num, np.cumsum(weights))
                pars[:] = pars[:,idx]
                weights[:] = weights[idx]
                weights[:] = weights / np.sum(weights)

        return s2


    def smoothing(self, s, lag=5, hook=None):
        threshold = self.num * self.ess_threshold_ratio
        pars = self.pars
        weights = self.weights
        num = self.num

        pars_hist = np.empty([pars.shape[0], pars.shape[1], lag]) * np.nan

        n = s.shape[1]
        s2 = np.empty(s.shape).astype(self.dtype)

        def h(pf, y): pass
        hook = hook if hook is not None else h

        for i in range(n):
            y = s[:,i].reshape(-1,1)
            hook(self, y)

            self.update(pars)
            weights[:] = np.exp( np.log(weights) + self._loglikelihood(y, pars) )
            weights[:] = weights / np.sum(weights)

            pars_hist[:,:,1:] = pars_hist[:,:,:-1].copy()
            pars_hist[:,:,0] = pars.copy()

            if lag<i:
                s2[:,i-lag] = self.estimate(pars_hist[:,:,-1], weights).ravel()

            if n-1==i:
                for j in range(lag):
                    s2[:,n-1-j] = self.estimate(pars_hist[:,:,-j], weights).ravel()

            ess = 1 / np.sum(weights**2)
            if ess < threshold:
                idx = resample(num, np.cumsum(weights))
                pars[:] = pars[:,idx]
                pars_hist[:] = pars_hist[:,idx,:]
                weights[:] = weights[idx]
                weights[:] = weights / np.sum(weights)

        return s2


@jit(i2[:](i2,f8[:]))
def resample(num, wcum):
    start = 0
    idxs = np.zeros(num).astype(np.int16)
    rands = np.sort(np.random.rand(num)).astype(np.float64)
    length = rands.size

    for i in range(length):
        for j in range(start, num):
            if rands[i] <= wcum[j]:
                idxs[i] = start = j
                break

    return idxs


def cauchy_noise(gma, num):
    return gma * np.tan(np.pi * (random.rand(gma.shape[0], num) - 1/2))

def normal_noise(sgm, num):
    return sgm * random.randn(sgm.shape[0], num)

def cauchy_logpdf(gma, x):
    return np.log( gma / np.pi ) - np.log(x**2 + gma**2)

def normal_logpdf(sgm, x):
    return - np.log( 2*np.pi ) /2 - np.log(sgm) - (x**2) / (2 * sgm**2)

def vonmises_logpdf(beta, x):
    return beta * np.cos(x) - np.log(2*np.pi*special.iv(0, beta))
