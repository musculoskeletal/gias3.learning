"""
FILE: kernelregression.py
LAST MODIFIED: 24-12-2015 
DESCRIPTION: Module for Gaussian kernel regression

===============================================================================
This file is part of GIAS2. (https://bitbucket.org/jangle/gias2)

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.
===============================================================================
"""
import logging

import numpy as np

log = logging.getLogger(__name__)


def _gaussianKernel(T, t, s):
    c1 = 1.0 / (s * np.sqrt(2 * np.pi))
    c2 = np.exp(-(T - t) ** 2.0 / (2 * s * s))
    return c1 * c2


def _gaussianKernelMax(s):
    return 1.0 / (s * np.sqrt(2 * np.pi))


def _weightedMean(x, w):
    # print x
    # print x.shape
    # print w.shape
    return (x * w).sum(-1) / w.sum()


def _weightedSD(x, w):
    u = _weightedMean(x, w)
    d = ((x.T - u).T) ** 2.0
    return np.sqrt((d * w).sum(-1) / w.sum())


class KernelRegressor(object):
    sigmaeps = 1e-6

    def __init__(self, k=2, sigma0=1.0, wmin=0.35):
        self.k = k
        self.sigma0 = sigma0
        self.wmin = wmin  # kernel weight cutoff

        self.nTarg = None  # target number of obs at each time point
        self.sigmas = None
        self.n = None
        self.y = None
        self.ytmeans = None
        self.ytSDs = None
        self.ytweights = None
        self.ytinds = None
        self.x = None
        self.xsamples = None
        self.xt = None
        self.xtn = None
        self.xtnTarg = None
        self.xmin = None
        self.xmax = None

    def fit(self, x, y, xmin, xmax, xsamples):
        """
        x is a 1-d array, the independent variable e.g. time

        if y is multivariate, each observation should be a column, so variables
        in rows.
        """
        self.x = x
        self.y = y
        self.xmin = xmin
        self.xmax = xmax
        self.sigmaeps = 1e-3 * (xmax - xmin)
        self.xsamples = xsamples
        self.xt = np.linspace(self.xmin, self.xmax, self.xsamples)

        self._fitInit()
        self._optWidths()

        return self.ytmeans

    def _fitInit(self):

        # initialise kernel width at each time point
        self.sigmas = np.ones(self.xsamples) * self.sigma0

        # calc number of obs at each y sampling (yt[i])
        self.xtn = []
        self.ytmeans = []
        self.ytweights = []
        self.ytSDs = []
        self.ytinds = []
        for t, s in zip(self.xt, self.sigmas):
            ty, tw, tyi = self._getKernelY(t, s)
            self.xtn.append(len(tw))
            self.ytweights.append(tw)
            self.ytinds.append(tyi)
            self.ytmeans.append(_weightedMean(ty, tw))
            self.ytSDs.append(_weightedSD(ty, tw))

        self.ytmeans = np.array(self.ytmeans)

        # calculate target number obs per time point
        self.xtnTarg = np.median(self.xtn)

    def _optWidths(self):
        for i in range(self.xsamples):
            tyi = None
            change = 0
            if self.xtn[i] > (self.xtnTarg + self.k):
                while self.xtn[i] > (self.xtnTarg + self.k):
                    self.sigmas[i] -= self.sigmaeps
                    ty, tw, tyi = self._getKernelY(self.xt[i], self.sigmas[i])
                    self.xtn[i] = len(tw)
                change = 1

            if self.xtn[i] < (self.xtnTarg - self.k):
                while self.xtn[i] < (self.xtnTarg - self.k):
                    self.sigmas[i] += self.sigmaeps
                    ty, tw, tyi = self._getKernelY(self.xt[i], self.sigmas[i])
                    self.xtn[i] = len(tw)
                change = 1

            # calculated weighted mean
            # if tyi is not None:
            #   print self.xt[i], tyi
            # else:
            #   print self.xt[i], 'no opt'
            if change:
                self.ytmeans[i] = _weightedMean(ty, tw)
                self.ytSDs[i] = _weightedSD(ty, tw)
                self.ytweights[i] = tw
                self.ytinds[i] = tyi

    def _getKernelY(self, t, s):
        xw = _gaussianKernel(self.x, t, s)
        validMask = xw >= (self.wmin * _gaussianKernelMax(s))
        validW = xw[validMask]
        validY = self.y[:, validMask]
        return validY, validW, np.where(validMask)[0]


def test():
    ndata = 500
    x = 3.0 * np.random.rand(ndata)
    y = np.sin(x) + 0.1 * (np.random.rand(ndata) - 0.5)

    r = KernelRegressor(k=2, sigma0=0.05, wmin=0.35)
    r.fit(x, y, 0.1, 2.9, 10)

    log.debug('xt      sigma     n     y_mean')
    for xt, s, n, ym in zip(r.xt, r.sigmas, r.xtn, r.ytmeans):
        log.debug(('{:4.3f}   {:4.3f}   {:4d}    {:4.3f}').format(xt, s, n, ym))

    import matplotlib.pyplot as plt
    f = plt.figure()
    plt.scatter(x, y)
    plt.plot(r.xt, r.ytmeans, 'r')
    plt.plot(r.xt, r.ytmeans + r.ytSDs, 'g--')
    plt.plot(r.xt, r.ytmeans - r.ytSDs, 'g--')
    plt.show()


if __name__ == '__main__':
    test()
