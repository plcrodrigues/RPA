#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:33:59 2017

@author: coelhorp
"""

from collections import OrderedDict
import numpy as np

from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

from pyriemann.tangentspace import TangentSpace, tangent_space, untangent_space
from pyriemann.estimation import Covariances, XdawnCovariances
from pyriemann.utils.mean import mean_riemann, geodesic_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from pyriemann.utils.distance import distance_riemann
from pyriemann.clustering import Potato
from pyriemann.classification import MDM

from sklearn.externals import joblib
from . import manopt as manifoptim

def get_target_split_motorimagery(target, ncovs_train):

    target_train_idx = []
    for j in np.unique(target['labels']):
        sel = np.arange(np.sum(target['labels'] == j))
        np.random.shuffle(sel)
        target_train_idx.append(np.arange(len(target['labels']))[target['labels'] == j][sel[:ncovs_train]])
    target_train_idx = np.concatenate(target_train_idx)
    target_test_idx  = np.array([i for i in range(len(target['labels'])) if i not in target_train_idx])

    target_train = {}
    target_train['covs'] = target['covs'][target_train_idx]
    target_train['labels'] = target['labels'][target_train_idx]

    target_test = {}
    target_test['covs'] = target['covs'][target_test_idx]
    target_test['labels'] = target['labels'][target_test_idx]

    return target_train, target_test

def get_sourcetarget_split_motorimagery(source, target, ncovs_train):
    target_train, target_test = get_target_split_motorimagery(target, ncovs_train)
    return source, target_train, target_test

def parallel_transport_covariance_matrix(C, R):

    return np.dot(invsqrtm(R), np.dot(C, invsqrtm(R)))

def parallel_transport_covariances(C, R):

    Cprt = []
    for Ci, Ri in zip(C, R):
        Cprt.append(parallel_transport_covariance_matrix(Ci, Ri))
    return np.stack(Cprt)

def gen_mixing(m):
    return np.random.randn(m,m)

def transform_org2rct(source_org, target_org_train, target_org_test):
    '''
        transform from original matrices to the parallel transported ones
        in this case, we're transporting reference (resting state) to the Identity
        org (original) -> rct (parallel transport)
    '''

    source_rct = {}
    T_source = np.stack([mean_riemann(source_org['covs'])] * len(source_org['covs']))
    source_rct['covs'] = parallel_transport_covariances(source_org['covs'], T_source)
    source_rct['labels'] = source_org['labels']

    M_target = mean_riemann(target_org_train['covs'])

    target_rct_train = {}
    T_target_train = np.stack([M_target]*len(target_org_train['covs']))
    target_rct_train['covs'] = parallel_transport_covariances(target_org_train['covs'], T_target_train)
    target_rct_train['labels'] = target_org_train['labels']

    target_rct_test = {}
    T_target_test = np.stack([M_target]*len(target_org_test['covs']))
    target_rct_test['covs'] = parallel_transport_covariances(target_org_test['covs'], T_target_test)
    target_rct_test['labels'] = target_org_test['labels']

    return source_rct, target_rct_train, target_rct_test

def transform_rct2str(source, target_train, target_test, pcoeff=False):
    '''

    '''

    covs_source = source['covs']
    covs_target_train = target_train['covs']
    covs_target_test = target_test['covs']

    source_pow  = {}
    source_pow ['covs'] = source['covs']
    source_pow ['labels'] = source['labels']

    n = covs_source.shape[1]
    disp_source = np.sum([distance_riemann(covi, np.eye(n)) ** 2 for covi in covs_source]) / len(covs_source)
    disp_target = np.sum([distance_riemann(covi, np.eye(n)) ** 2 for covi in covs_target_train]) / len(covs_target_train)
    p = np.sqrt(disp_target / disp_source)

    target_pow_train  = {}
    target_pow_train['covs'] = np.stack([powm(covi, 1.0/p) for covi in covs_target_train])
    target_pow_train['labels'] = target_train['labels']

    target_pow_test  = {}
    target_pow_test['covs'] = np.stack([powm(covi, 1.0/p) for covi in covs_target_test])
    target_pow_test['labels'] = target_test['labels']

    if pcoeff:
        return source_pow , target_pow_train, target_pow_test, p
    else:
        return source_pow , target_pow_train, target_pow_test

def transform_rct2rot(source, target_train, target_test):
    '''
        rotate the re-centered matrices from the target so they match with those from source
        note that we use information from class labels of some of the target covariances (not all but some)
        rct (re-centered matrices) -> rot (rotated re-centered matrices)
    '''

    source_rot = {}
    source_rot['covs'] = source['covs']
    source_rot['labels'] = source['labels']

    target_rot_train = {}
    target_rot_train['labels'] = target_train['labels']

    target_rot_test = {}
    target_rot_test['labels'] = target_test['labels']

    M_source = []
    for i in np.unique(source['labels']):
        M_source_i = mean_riemann(source['covs'][source['labels'] == i])
        M_source.append(M_source_i)

    M_target_train = []
    for j in np.unique(target_train['labels']):
        M_target_train_j = mean_riemann(target_train['covs'][target_train['labels'] == j])
        M_target_train.append(M_target_train_j)

    R = manifoptim.get_rotation_matrix(M=M_source, Mtilde=M_target_train, dist='euc')

    covs_target_train = np.stack([np.dot(R, np.dot(covi, R.T)) for covi in target_train['covs']])
    target_rot_train['covs'] = covs_target_train

    covs_target_test = np.stack([np.dot(R, np.dot(covi, R.T)) for covi in target_test['covs']])
    target_rot_test['covs'] = covs_target_test

    return source_rot, target_rot_train, target_rot_test

def transform_str2rot(source, target_train, target_test):
    return transform_rct2rot(source, target_train, target_test)

def transform_org2talmon(source, target_train, target_test):

    covs_source = source['covs']
    covs_target_train = target_train['covs']
    covs_target_test = target_test['covs']

    M_source = mean_riemann(covs_source)
    M_target_train = mean_riemann(covs_target_train)
    M = geodesic_riemann(M_source, M_target_train, alpha=0.5)

    Gamma = ParallelTransport(reference_old=M_source,reference_new=M)
    covs_source_transp = Gamma.fit_transform(covs_source)

    Gamma = ParallelTransport(reference_old=M_target_train,reference_new=M)
    covs_target_train_transp = Gamma.fit_transform(covs_target_train)
    covs_target_test_transp = Gamma.transform(covs_target_test)

    source_talmon = {}
    source_talmon['labels'] = source['labels']
    source_talmon['covs'] = covs_source_transp

    target_talmon_train = {}
    target_talmon_train['labels'] = target_train['labels']
    target_talmon_train['covs'] = covs_target_train_transp

    target_talmon_test = {}
    target_talmon_test['labels'] = target_test['labels']
    target_talmon_test['covs'] = covs_target_test_transp

    return source_talmon, target_talmon_train, target_talmon_test

class ParallelTransport(BaseEstimator, TransformerMixin):
    '''Parallel Transport in the SPD Manifold with AIRM distance

    Parallel transport along the geodesic joining two SPD matrices (ref_old and ref_new)
    The input and output can be given in terms of a set of covariance matrices or tangent vectors

    Parameters
    ----------
    reference_old: ndarray, shape (n_channels, n_channels)
        covariance matrix of the old reference for tangent vectors
    reference_new: ndarray, shape (n_channels, n_channels)
        covariance matrix of the new reference for tangent vectors
    tangent_old: boolean (default: False)
        whether the old matrices should be given as tangent vectors or not
    tangent_new: boolean (default: False)
        whether the new matrices should be given as tangent vectors or not

    '''

    def __init__(self, reference_old, reference_new, tangent_old=False, tangent_new=False):
        self.reference_old = reference_old
        self.reference_new = reference_new
        self.tangent_old = tangent_old
        self.tangent_new = tangent_new

    def fit(self, X, y=None):
        """Fit.

        Calculate the matrix used for the parallel transport

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels) or (n_trials, n_channels*(n_channels+1)/2)
            ndarray of trials.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial, not used.

        Returns
        -------
        self : ParallelTransport instance
            The ParallelTransport instance.
        """
        self._fit(X)
        return self

    def transform(self, X, y=None):
        """Transport the tangent vectors to a new reference matrix.
        X is the old set of matrices.

        Parameters
        ----------
        X : ndarray, shape (n_trials, n_channels, n_channels) or (n_trials, n_channels*(n_channels+1)/2)
            ndarray of trials.

        Returns
        -------
        Xnew : ndarray, shape (n_trials, n_channels, n_channels) or (n_trials, n_channels*(n_channels+1)/2)
            ndarray of transported covariance matrices or tangent vectors for each trial.
        """
        # X is a matrix with the covariances to be transported
        # 'transform' will do the parallel transport of the cov matrices
        Xnew = self._transform(X)
        return Xnew

    def _fit(self, X):
        Ri = self.reference_old
        Rf = self.reference_new
        A = sqrtm(Ri)
        B = sqrtm(np.dot(invsqrtm(Ri), np.dot(Rf, invsqrtm(Ri))))
        C = invsqrtm(Ri)
        W = np.dot(A, np.dot(B, C))
        self.transporter_ = W

    def _transform(self, X):

        W = self.transporter_
        Ri = self.reference_old
        Rf = self.reference_new
        Nt = X.shape[0]

        # detect which kind of input : tangent vectors or cov matrices
        if self.tangent_old:
            # if tangent vectors are given, transform them back to covs
            # (easier to have tg vectors in the form of symmetric matrices later)
            X = untangent_space(X, Ri)

        # transform covariances to their tangent vectors with respect to Ri
        # (these tangent vectors are in the form of symmetric matrices)
        eta_i = np.zeros(X.shape)
        Ri_sqrt = sqrtm(Ri)
        Ri_invsqrt = invsqrtm(Ri)
        for i in range(Nt):
            Li = logm(np.dot(Ri_invsqrt, np.dot(X[i], Ri_invsqrt)))
            eta_i[i,:,:] = np.dot(Ri_sqrt, np.dot(Li, Ri_sqrt))

        # multiply the tangent vectors by the transport matrix W
        eta_f = np.zeros(X.shape)
        for i in range(Nt):
            eta_f[i,:,:] = np.dot(W, np.dot(eta_i[i], W.T))

        # transform tangent vectors to covariance matrices with respect to Rf
        Xnew = np.zeros(X.shape)
        Rf_sqrt = sqrtm(Rf)
        Rf_invsqrt = invsqrtm(Rf)
        for i in range(Nt):
            Ef = expm(np.dot(Rf_invsqrt, np.dot(eta_f[i], Rf_invsqrt)))
            Xnew[i,:,:] = np.dot(Rf_sqrt, np.dot(Ef, Rf_sqrt))

        # transform back to tangent vectors (flat form, not sym matrix) if needed
        if self.tangent_new:
            Xnew = tangent_space(Xnew, Rf)

        return Xnew
