import re, time
import sys

import numpy as n
from scipy.special import gammaln, psi

n.random.seed(100000001)
"""
Created : 03 /08 /2016
@author : Ha Nhat Cuong

"""

def dirichlet_expectation(alpha):
    if len(alpha.shape) == 1:
        return psi(alpha) - psi(n.sum(alpha))
    return psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis]


class BatchVB:

    def __init__(self, num_terms, num_topics, iter_infer,
                 meanchangethresh, alpha, eta):
        #self._D = num_docs

        self._K = num_topics
        self._W = num_terms

        self._iter_infer = iter_infer
        self._meanchangethresh = meanchangethresh

        self._alpha = alpha
        self._eta = eta

        self._lambda = 1*n.random.gamma(100.0, 1.0/100.0, (self._K, self._W))
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)

    def do_e_step(self, wordids, wordcts):
        batchD = len(wordids)

        gamma = n.random.gamma(100.0, 1.0/100.0, (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._lambda.shape)

        for d in range(0, batchD) :
            ids = wordids[d]
            cts = wordcts[d]   # 1 * size(ids)

            gammad = gamma[d, :]  # gammad [1* K]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]    #1*K
            expElogbetad = self._expElogbeta[:, ids]   # K * size(ids)

            phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100   # 1 * size(ids)

            for it in range(0, self._iter_infer):
                lastgamma = gammad

                gammad = self._alpha + expElogthetad*\
                    n.dot(cts/phinorm, expElogbetad.T)

                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100

                meanchange = n.mean(abs(lastgamma - gammad))
                if meanchange < self._meanchangethresh :
                    break

                gamma[d, :] = gammad
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm) # K  * size(ids)

        sstats = sstats * self._expElogbeta         # K * W
        return gamma, sstats  # batchD * K , K * W

    def update_lambda(self, sstats):

        self._lambda = self._eta + sstats # num_topics * W
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)


    def static_online(self, wordids, wordcts):
        # E step
        start = time.time()
        (gamma, sstats) = self.do_e_step(wordids, wordcts)
        end1 = time.time()
        # M step
        self.update_lambda(sstats)
        end2 = time.time()
        return (end1 - start, end2 - start, gamma)
