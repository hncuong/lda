import re
import time
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


class OnlineVB:

    def __init__(self, num_terms, num_topics, num_docs, batch_size,
                 iter_infer, alpha, eta, tau0, kappa):
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

        self._D = num_docs
        self._batchsize = batch_size

        self._K = num_topics
        self._W = num_terms

        self._iter_infer = iter_infer
        # self._meanchangethresh = meanchangethresh

        self._alpha = alpha
        self._eta = eta

        self._tau0 = tau0
        self._kappa = kappa
        self._updatect = 1

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
                # lastgamma = gammad

                gammad = self._alpha + expElogthetad*\
                    n.dot(cts/phinorm, expElogbetad.T)

                Elogthetad = dirichlet_expectation(gammad)
                expElogthetad = n.exp(Elogthetad)
                phinorm = n.dot(expElogthetad, expElogbetad) + 1e-100

                # meanchange = n.mean(abs(lastgamma - gammad))
                # if meanchange < self._meanchangethresh :
                #     break

                gamma[d, :] = gammad
            sstats[:, ids] += n.outer(expElogthetad.T, cts/phinorm) # K  * size(ids)

        sstats = sstats * self._expElogbeta         # K * W
        return  ((gamma, sstats))

    def update_lambda(self, sstats, batch_size):

        rhot = pow(self._tau0 + self._updatect, -self._kappa)
        self._lambda = (1 - rhot) * self._lambda + \
            rhot * (self._eta + self._D * sstats / batch_size)
        self._Elogbeta = dirichlet_expectation(self._lambda)
        self._expElogbeta = n.exp(self._Elogbeta)
        self._updatect += 1

    def static_online(self, wordids, wordcts):
        #E step
        start = time.time()
        (gamma, sstats) = self.do_e_step(wordids, wordcts)
        end1 = time.time()
        #M step
        self.update_lambda(sstats, len(wordids))
        end2 = time.time()
        return (end1 - start, end2 - start, gamma)
