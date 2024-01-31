#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:33:59 2017
@author: coelhorp

Modified on Dec 2 2023
@author: Carl
"""

from collections import OrderedDict
import numpy as np

from scipy.linalg import eigh
from sklearn.base import BaseEstimator, TransformerMixin

from pyriemann.tangentspace import TangentSpace, tangent_space, untangent_space
from pyriemann.estimation import Covariances, XdawnCovariances, ERPCovariances
from pyriemann.utils.mean import mean_riemann, geodesic_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from pyriemann.utils.distance import distance_riemann
from pyriemann.classification import MDM

import joblib
from . import manopt as manifoptim

def get_target_split_motorimagery(target, ncovs_train):

    nclasses = len(np.unique(target['labels']))

    if not(hasattr(ncovs_train, "__len__")):
        ncovs_train = [ncovs_train] * nclasses

    target_train_idx = []
    for j, label in enumerate(np.unique(target['labels'])):
        sel = np.arange(np.sum(target['labels'] == label))
        np.random.shuffle(sel)
        target_train_idx.append(np.arange(len(target['labels']))[target['labels'] == label][sel[:ncovs_train[j]]])
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

def get_sourcetarget_split_p300(source, target, ncovs_train):

    X_source = source['epochs']
    y_source = source['labels'].flatten()
    covs_source = ERPCovariances(classes=[2], estimator='lwf').fit_transform(X_source, y_source)

    source = {}
    source['covs'] = covs_source
    source['labels'] = y_source

    X_target = target['epochs']
    y_target = target['labels'].flatten()

    sel = np.arange(len(y_target))
    np.random.shuffle(sel)
    X_target = X_target[sel]
    y_target = y_target[sel]

    idx_erps = np.where(y_target == 2)[0][:ncovs_train]
    idx_rest = np.where(y_target == 1)[0][:ncovs_train*5] # because there's one ERP in every 6 flashes

    idx_train = np.concatenate([idx_erps, idx_rest])
    idx_test  = np.array([i for i in range(len(y_target)) if i not in idx_train])

    erp = ERPCovariances(classes=[2], estimator='lwf')
    erp.fit(X_target[idx_train], y_target[idx_train])

    target_train = {}
    covs_target_train = erp.transform(X_target[idx_train])
    y_target_train = y_target[idx_train]
    target_train['covs'] = covs_target_train
    target_train['labels'] = y_target_train

    target_test = {}
    covs_target_test = erp.transform(X_target[idx_test])
    y_target_test = y_target[idx_test]
    target_test['covs'] = covs_target_test
    target_test['labels'] = y_target_test

    return source, target_train, target_test

def parallel_transport_covariance_matrix(C, R):

    return np.dot(invsqrtm(R), np.dot(C, invsqrtm(R)))

def parallel_transport_covariances(C, R):

    Cprt = []
    for Ci, Ri in zip(C, R):
        Cprt.append(parallel_transport_covariance_matrix(Ci, Ri))
    return np.stack(Cprt)

def recenter(covs, M=None, sample_weights=None, return_M=False):
    '''Parallel transport covariance matrices to reference to the Identity matrix

    params:
        covs: n covariance matrices to be transformed
        M: the original reference (mean) of covs. If not provided, Riemannian geometric mean of covs will be computed
        sample_weights: 1d array of size n to weight the covs during M computation
    '''
    n_covs = len(covs)

    if sample_weights is None:
        sample_weights = np.ones(n_covs)
    if M is None:
        M = mean_riemann(covs, sample_weight=sample_weights)
    
    _M = np.stack([M] * n_covs)
    if return_M:
        return parallel_transport_covariances(covs, _M), M
    else:
        return parallel_transport_covariances(covs, _M)


def transform_org2rct(source_org, target_org_train, target_org_test, source_weights=None, target_train_weights=None):
    '''
        transform from original matrices to the parallel transported ones
        in this case, we're transporting reference (resting state) to the Identity
        org (original) -> rct (parallel transport)

        params:
            source_weights: 1d array with the same length as no. matrices in source_org
            target_train_weights: 1d array with the same length as no. matrices in target_org_train
    '''

    source_rct = {}
    source_rct['covs'] = recenter(source_org['covs'], sample_weights=source_weights, return_M=False)
    source_rct['labels'] = source_org['labels']

    covs_target_train, M_target = recenter(target_org_train['covs'], sample_weights=target_train_weights, return_M=True)

    target_rct_train = {}
    target_rct_train['covs'] = covs_target_train
    target_rct_train['labels'] = target_org_train['labels']

    target_rct_test = {}
    target_rct_test['covs'] = recenter(target_org_test['covs'], M=M_target, return_M=False)
    target_rct_test['labels'] = target_org_test['labels']

    return source_rct, target_rct_train, target_rct_test


def transform_org2rct_p300(source, target_train, target_test, weight_samples=False):
    source_weights = np.ones(len(source['labels']))
    if weight_samples:
        source_weights[source['labels'] == 2] = 5
    
    target_train_weights = np.ones(len(target_train['labels']))
    if weight_samples:
        target_train_weights[target_train['labels'] == 2] = 5
    
    return transform_org2rct(source, target_train, target_test, 
                             source_weights=source_weights, target_train_weights=target_train_weights)


def compute_pcoeff(covs_source, covs_target):
    '''Compute p coefficient as sqrt(mean_target_distances/mean_source_distances)'''
    n = covs_source.shape[1]
    disp_source = np.sum([distance_riemann(covi, np.eye(n)) ** 2 for covi in covs_source]) / len(covs_source)
    disp_target = np.sum([distance_riemann(covi, np.eye(n)) ** 2 for covi in covs_target]) / len(covs_target)

    return np.sqrt(disp_target / disp_source)


def stretch(covs, p):
    return np.stack([powm(covi, 1.0/p) for covi in covs])


def unit_stretch(covs, return_p=False):
    n_covs, n_ch, _ = covs.shape
    I = np.repeat(np.eye(n_ch)[np.newaxis,...], n_covs, axis=0)
    disp = np.sum(distance_riemann(covs, I) ** 2) / n_covs
    p = np.sqrt(disp/1.)

    if return_p:
        return powm(covs, 1./p), p
    else:
        return powm(covs, 1./p)


def transform_rct2str(source, target_train, target_test, pcoeff=False, transform_target=True):
    '''

    '''
    covs_source = source['covs']
    covs_target_train = target_train['covs']
    covs_target_test = target_test['covs']

    source_pow  = {}
    source_pow ['covs'] = source['covs']
    source_pow ['labels'] = source['labels']

    target_pow_train  = {}
    target_pow_test  = {}
    target_pow_train['labels'] = target_train['labels']
    target_pow_test['labels'] = target_test['labels']

    if transform_target:
        p = compute_pcoeff(covs_source, covs_target_train) 
        target_pow_train['covs'] = stretch(covs_target_train, p)
        target_pow_test['covs'] = stretch(covs_target_test, p)
    else:
        p = compute_pcoeff(covs_target_train, covs_source)
        source_pow['covs'] = stretch(source_pow['covs'], p)
        target_pow_train['covs'] = covs_target_train
        target_pow_test['covs'] = covs_target_test

    if pcoeff:
        return source_pow , target_pow_train, target_pow_test, p
    else:
        return source_pow , target_pow_train, target_pow_test


def compute_R(covs_source, covs_target, labels_source, labels_target, weights=None, distance='euc'):
    '''Compute the rotation matrix to be applied to the target'''
    M_source = [mean_riemann(covs_source[labels_source == i]) for i in np.unique(labels_source)]
    M_target = [mean_riemann(covs_target[labels_target == j]) for j in np.unique(labels_target)]
    R = manifoptim.get_rotation_matrix(M=M_source, Mtilde=M_target, weights=weights, dist=distance)
    return R


def rotate(covs, R):
    return np.stack([np.dot(R, np.dot(covi, R.T)) for covi in covs])


def transform_rct2rot(source, target_train, target_test, weights=None, distance='euc', transform_target=True):
    '''
        if transform_target: rotate the re-centered matrices from the target so they match with those from source
        else: rotate that from the source so they match with those from target_train
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

    if transform_target:
        R = compute_R(source['covs'], target_train['covs'], source['labels'], target_train['labels'], 
                      weights=weights, distance=distance)
        target_rot_train['covs'] = rotate(target_train['covs'], R)
        target_rot_test['covs'] = rotate(target_test['covs'], R)
    else:
        R = compute_R(target_train['covs'], source['covs'], target_train['labels'], source['labels'], 
                      weights=weights, distance=distance)
        source_rot['covs'] = rotate(source_rot['covs'], R)
        target_rot_train['covs'] = target_train['covs']
        target_rot_test['covs'] = target_test['covs']

    return source_rot, target_rot_train, target_rot_test


def transform_str2rot(source, target_train, target_test, **kwargs):
    return transform_rct2rot(source, target_train, target_test, **kwargs)


def transform_rct2rot_p300(source, target_train, target_test, class_weights, distance='euc', transform_target=True):
    return transform_rct2rot(source, target_train, target_test, weights=class_weights, 
                             distance=distance, transform_target=transform_target)


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

