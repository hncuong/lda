# -*- coding: utf-8 -*-
"""
Created on Sat May 31 20:38:39 2014

@author: doanphongtung
"""

import time
import numpy as np

def alpha_gradient_search(cts, beta_x, x, alpha):
    left = 0.
    right = alpha
    for j in xrange(10):
        alpha = (left + right) / 2.
        df = (cts * beta_x) / (alpha * beta_x + x)
        df = sum(df)
        if abs(df) < 1e-10:
            break
        if df < 0:
            right = alpha
        else:
            left = alpha
    return(alpha)

class StreamingFW:
    
    def __init__(self, num_terms, num_topics, conv_infer, iter_infer):              
        self.num_topics = num_topics
        self.num_terms = num_terms
        
        self.beta = np.random.rand(self.num_topics, self.num_terms) + 1e-10
        # normalize beta
        beta_norm = self.beta.sum(axis = 1)
        self.beta /= beta_norm[:, np.newaxis]
        
        self.logbeta = np.log(self.beta)
        
        self.theta_init = [1e-10] * num_topics
        self.theta_vert = 1. - 1e-10 * num_topics
        
        self.INF_MAX_CONV = conv_infer
        self.INF_MAX_ITER = iter_infer
    
    def e_step(self, batch_size, wordids, wordcts):
        # declare theta of minibatch and list of nonzero indexes
        theta = np.zeros((batch_size, self.num_topics))
        index = [{} for d in xrange(batch_size)]
        # inference
        for d in xrange(batch_size):
            (thetad, indexd) = self.infer_doc(wordids[d], wordcts[d])
            theta[d,:] = thetad
            index[d] = indexd
        return(theta, index)
        
    """
    f = log P(d) = sum(over j in I_d) d_j * log(x_j)
    """
    def infer_doc(self, ids, cts):
        # locate cache memory
        beta = self.beta[:,ids]
        logbeta = self.logbeta[:,ids]
        nonzero = set()
        #initialize theta
        theta = np.array(self.theta_init)
        f = np.dot(logbeta, cts)
        index = np.argmax(f); nonzero.add(index)
        theta[index] = self.theta_vert
        x = np.copy(beta[index,:])
        alpha = 1.
        # loop to find the projection
        lkh_old = 1e10
        for l in xrange(1,self.INF_MAX_ITER):
            # select direction and update
            df = np.dot(beta, cts / x)
            index = np.argmax(df); nonzero.add(index)
            beta_x = beta[index,:] - x
            alpha = alpha_gradient_search(cts, beta_x, x, alpha)
            if alpha < 1e-10:
                break
            theta *= 1 - alpha
            theta[index] += alpha
            x += alpha * (beta_x)
            # convergence check
            logx = np.log(x)
            lkh = np.dot(cts, logx)
            converge = (lkh_old - lkh) / lkh_old
            if converge < self.INF_MAX_CONV:
                break
            else:
                lkh_old = lkh
        return(theta, np.array(list(nonzero)))
    
    """
    Online
    """
    def m_step(self, batch_size, wordids, wordcts, theta, index):
        # compute sufficient sstatistics
        sstats = np.zeros((self.num_topics, self.num_terms))
        for d in xrange(batch_size):
            phi_d = self.beta[index[d], :]
            phi_d = phi_d[:, wordids[d]]
            theta_d = theta[d, index[d]]
            phi_d *= theta_d[:, np.newaxis]
            phi_norm = phi_d.sum(axis = 0)
            phi_d *= (wordcts[d] / phi_norm)
            for i in xrange(len(index[d])):
                sstats[index[d][i], wordids[d]] += phi_d[i, :]
        # update
        self.beta += sstats
        # normalize beta
        beta_norm = self.beta.sum(axis = 1)
        self.beta /= beta_norm[:, np.newaxis]
        self.logbeta = np.log(self.beta)        
        
    def static_online(self, batch_size, wordids, wordcts):
        # E step
        start1 = time.time()
        (theta, index) = self.e_step(batch_size, wordids, wordcts)
        end1 = time.time()
        # M step
        start2 = time.time()
        self.m_step(batch_size, wordids, wordcts, theta, index)
        end2 = time.time()
        return(end1 - start1, end2 - start2, theta)
