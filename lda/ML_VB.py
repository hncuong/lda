import re
import time
import sys

import numpy as n
from scipy.special import gammaln, psi

n.random.seed(100000001)
"""
Created : 23 /09 /2016
@author : Ha Nhat Cuong

"""


def dirichlet_expectation(alpha):
    if len(alpha.shape) == 1:
        return psi(alpha) - psi(n.sum(alpha))
    return psi(alpha) - psi(n.sum(alpha, 1))[:, n.newaxis]


class MLVB:

    def __init__(self, num_terms, num_topics,
                 iter_infer, alpha, tau0, kappa):
        """
        Arguments:
        K: Number of topics
        vocab: A set of words to recognize. When analyzing documents, any word
           not in this set will be ignored.
        D: Total number of documents in the population. For a fixed corpus,
           this is the size of the corpus. In the truly online setting, this
           can be an estimate of the maximum number of documents that
           could ever be seen.
        alpha: Hyperparameter for prior on weight vectors theta
        eta: Hyperparameter for prior on topics beta
        tau0: A (positive) learning parameter that downweights early iterations
        kappa: Learning rate: exponential decay rate---should be between
             (0.5, 1.0] to guarantee asymptotic convergence.

        Note that if you pass the same set of D documents in every time and
        set kappa=0 this class can also be used to do batch VB.
        """

        self._K = num_topics
        self._W = num_terms

        self._iter_infer = iter_infer

        self._alpha = alpha

        self._tau0 = tau0
        self._kappa = kappa
        self._updatect = 1

        self._beta = 1*n.random.gamma(100.0, 1.0/100.0, (self._K, self._W))
        beta_norm = self._beta.sum(axis = 1)
        self._beta /= beta_norm[:, n.newaxis]

    def do_e_step(self, wordids, wordcts):
        batchD = len(wordids)

        gamma = n.random.gamma(100.0, 1.0/100.0, (batchD, self._K))
        Elogtheta = dirichlet_expectation(gamma)
        expElogtheta = n.exp(Elogtheta)

        sstats = n.zeros(self._beta.shape)

        for d in range(0, batchD) :
            ids = wordids[d]
            cts = wordcts[d]   # 1 * size(ids)

            gammad = gamma[d, :]  # gammad [1* K]
            Elogthetad = Elogtheta[d, :]
            expElogthetad = expElogtheta[d, :]    #1*K
            # expElogbetad = self._expElogbeta[:, ids]   # K * size(ids)
            betad = self._beta[:, ids] # K * size(ids)

            phinorm = n.dot(expElogthetad, betad) + 1e-100   # 1 * size(ids)

            for it in range(0, self._iter_infer):
                gammad = self._alpha + expElogthetad*\
                    n.dot(cts/phinorm, betad.T)

                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, betad) + 1e-100

                gamma[d, :] = gammad
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm) # K  * size(ids)

        sstats = sstats * self._beta         # K * W
        return  ((gamma, sstats))

    def update_beta(self, sstats):

        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        beta_norm = sstats.sum(axis = 1)
        beta = sstats / beta_norm[:, n.newaxis]
        self._beta = (1 - rhot) * self._beta + rhot * beta
        self._updatect += 1

    def static_online(self, wordids, wordcts):
        #E step
        start = time.time()
        (gamma, sstats) = self.do_e_step(wordids, wordcts)
        end1 = time.time()
        #M step
        self.update_beta(sstats)
        end2 = time.time()
        return (end1 - start, end2 - start, gamma)
