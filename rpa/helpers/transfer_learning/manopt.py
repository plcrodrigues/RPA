#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 15:25:54 2017

@author: coelhorp
"""

from scipy.linalg import eigh
import autograd.numpy as np
from pymanopt.manifolds import Rotations
from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, logm

from functools import partial

def gen_symm(n):
    A = np.random.randn(n,n)
    return A + A.T

def gen_spd(n):
    A = gen_symm(n)
    w,v = eigh(A)
    w = np.diag(np.random.rand(len(w)))
    return np.dot(v, np.dot(w, v.T))

def gen_anti(n):
    A = np.random.randn(n,n)
    return A - A.T

def gen_orth(n):
    A = gen_symm(n)
    _,Q = eigh(A)
    return Q

def cost_function_pair_euc(M, Mtilde, Q):
    t1 = M
    t2 = np.dot(Q, np.dot(Mtilde, Q.T))
    return np.linalg.norm(t1 - t2)**2

def cost_function_pair_rie(M, Mtilde, Q):
    t1 = M
    t2 = np.dot(Q, np.dot(Mtilde, Q.T))
    return distance_riemann(t1, t2)**2

def cost_function_full(Q, M, Mtilde, weights=None, dist=None):
    if weights is None:
        weights = np.ones(len(M)) 
    else:
        weights = np.array(weights)
        
    if dist is None:
        dist = 'euc'
        
    cost_function_pair = {}
    cost_function_pair['euc'] = cost_function_pair_euc
    cost_function_pair['rie'] = cost_function_pair_rie    
        
    c = []
    for Mi, Mitilde in zip(M, Mtilde):
        ci = cost_function_pair[dist](Mi, Mitilde, Q)
        c.append(ci)
    c = np.array(c)
    
    return np.dot(c, weights)

def egrad_function_pair_rie(M, Mtilde, Q):
    Mtilde_invsqrt = invsqrtm(Mtilde)
    M_sqrt = sqrtm(M)
    term_aux = np.dot(Q, np.dot(M, Q.T))
    term_aux = np.dot(Mtilde_invsqrt, np.dot(term_aux, Mtilde_invsqrt))
    return 4 * np.dot(np.dot(Mtilde_invsqrt, logm(term_aux)), np.dot(M_sqrt, Q))

def egrad_function_full_rie(Q, M, Mtilde, weights=None):

    if weights is None:
        weights = np.ones(len(M)) 
    else:
        weights = np.array(weights)

    g = []
    for Mi, Mitilde, wi in zip(M, Mtilde, weights):
        gi = egrad_function_pair_rie(Mi, Mitilde, Q)
        g.append(gi * wi)
    g = np.sum(g, axis=0)        
    
    return g

def get_rotation_matrix(M, Mtilde, weights=None, dist=None):
    
    if dist is None:
        dist = 'euc'
    
    n = M[0].shape[0]
        
    # (1) Instantiate a manifold
    manifold = Rotations(n)
    
    # (2) Define cost function and a problem
    if dist == 'euc':
        cost = partial(cost_function_full, M=M, Mtilde=Mtilde, weights=weights, dist=dist)    
        problem = Problem(manifold=manifold, cost=cost, verbosity=0)
    elif dist == 'rie':
        cost = partial(cost_function_full, M=M, Mtilde=Mtilde, weights=weights, dist=dist)    
        egrad = partial(egrad_function_full_rie, M=M, Mtilde=Mtilde, weights=weights) 
        problem = Problem(manifold=manifold, cost=cost, egrad=egrad, verbosity=0)
        
    # (3) Instantiate a Pymanopt solver
    solver = SteepestDescent(mingradnorm=1e-3)   
    
    # let Pymanopt do the rest
    Q_opt = solver.solve(problem)    
    
    return Q_opt
    














