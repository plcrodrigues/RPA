#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 18:59:19 2017

@author: coelhorp
"""

import numpy as np

def make_distanceMatrix(points, distance):
    Npoints = len(points)
    distmatrix = np.zeros((Npoints, Npoints))
    for ii,pi in enumerate(points):
        for jj,pj in enumerate(points):
            distmatrix[ii,jj] = distance(pi,pj)            
            
    return distmatrix

def renormalize_kernel(kernel, alpha):    
    q = np.power(np.dot(kernel, np.ones(len(kernel))), alpha)
    K = np.divide(kernel, np.outer(q,q))    
    return K        

def make_kernelMatrix(distmatrix, eps):
    kernel = np.exp(-distmatrix**2/eps)            
    return kernel

def make_transitionMatrix(kernel):    
    d = np.sqrt(np.dot(kernel, np.ones(len(kernel))))
    P = np.divide(kernel, np.outer(d, d))    
    return P

def get_diffusionEmbedding(points=[], distance=[], distmatrix=None, alpha=1.0, tdiff=0, eps=None):
    
    if distmatrix is None:
        d = make_distanceMatrix(points, distance)          
    else:
        d = distmatrix

    if eps is None:
        # using heuristic from the R package for diffusion maps
        eps = 2*np.median(d)**2      
            
    K = make_kernelMatrix(distmatrix=d, eps=eps)
    Kr = renormalize_kernel(K, alpha=alpha)            
    P = make_transitionMatrix(Kr)
    u,s,v = np.linalg.svd(P)    
    
    phi = np.copy(u)
    for i in range(len(u)):
        phi[:,i] = (s[i]**tdiff)*np.divide(u[:,i], u[:,0])
    
    return phi, s

    
    
  