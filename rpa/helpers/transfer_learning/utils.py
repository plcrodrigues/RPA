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
from pyriemann.estimation import Covariances, XdawnCovariances, ERPCovariances
from pyriemann.utils.mean import mean_riemann, geodesic_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from pyriemann.utils.distance import distance_riemann
from pyriemann.clustering import Potato
from pyriemann.classification import MDM

from ot.da import distribution_estimation_uniform
from ot.da import sinkhorn_lpl1_mm

from sklearn.externals import joblib
from . import manopt as manifoptim
from riemann_lab import get_datasets as GD

# def get_target_split_motorimagery(target, ncovs_train):

#     target_train_idx = []
#     for j in np.unique(target['labels']):
#         sel = np.arange(np.sum(target['labels'] == j))
#         np.random.shuffle(sel)
#         target_train_idx.append(np.arange(len(target['labels']))[target['labels'] == j][sel[:ncovs_train]])
#     target_train_idx = np.concatenate(target_train_idx)
#     target_test_idx  = np.array([i for i in range(len(target['labels'])) if i not in target_train_idx])

#     target_train = {}
#     target_train['covs'] = target['covs'][target_train_idx]
#     target_train['labels'] = target['labels'][target_train_idx]

#     target_test = {}
#     target_test['covs'] = target['covs'][target_test_idx]
#     target_test['labels'] = target['labels'][target_test_idx]

#     return target_train, target_test

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

def transform_org2rct_p300(source, target_train, target_test, weight_samples=False):

    source_rct = {}
    source_rct['labels'] = source['labels']
    weights = np.ones(len(source['labels']))
    if weight_samples:
        weights[source['labels'] == 2] = 5
    T = mean_riemann(source['covs'], sample_weight=weights)
    T_source = np.stack([T]*len(source['covs']))
    source_rct['covs'] = parallel_transport_covariances(source['covs'], T_source)

    target_rct_train = {}
    target_rct_train['labels'] = target_train['labels']
    weights = np.ones(len(target_train['labels']))
    if weight_samples:
        weights[target_train['labels'] == 2] = 5
    M_train = mean_riemann(target_train['covs'], sample_weight=weights)
    T_target = np.stack([M_train]*len(target_train['covs']))
    target_rct_train['covs'] = parallel_transport_covariances(target_train['covs'], T_target)

    target_rct_test = {}
    target_rct_test['labels'] = target_test['labels']
    M_test = M_train
    T_target = np.stack([M_test]*len(target_test['covs']))
    target_rct_test['covs'] = parallel_transport_covariances(target_test['covs'], T_target)

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

def transform_rct2rot(source, target_train, target_test, weights=None, distance='euc'):
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

    R = manifoptim.get_rotation_matrix(M=M_source, Mtilde=M_target_train, weights=weights, dist=distance)

    covs_target_train = np.stack([np.dot(R, np.dot(covi, R.T)) for covi in target_train['covs']])
    target_rot_train['covs'] = covs_target_train

    covs_target_test = np.stack([np.dot(R, np.dot(covi, R.T)) for covi in target_test['covs']])
    target_rot_test['covs'] = covs_target_test

    return source_rot, target_rot_train, target_rot_test

def transform_str2rot(source, target_train, target_test):
    return transform_rct2rot(source, target_train, target_test)

def transform_rct2rot_p300(source, target_train, target_test, class_weights, distance='euc'):

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

    #R = manifoptim.get_rotation_matrix(M=M_source, Mtilde=M_target_train, dist='euc', weights=[1, 5])
    R = manifoptim.get_rotation_matrix(M=M_source, Mtilde=M_target_train, dist=distance, weights=class_weights)    

    covs_target_train = np.stack([np.dot(R, np.dot(covi, R.T)) for covi in target_train['covs']])
    target_rot_train['covs'] = covs_target_train

    covs_target_test = np.stack([np.dot(R, np.dot(covi, R.T)) for covi in target_test['covs']])
    target_rot_test['covs'] = covs_target_test

    return source_rot, target_rot_train, target_rot_test

def transform_org2opt(source, target_train, target_test):

    target_opt_train = {}
    target_opt_test = {}

    target_opt_train['labels'] = target_train['labels']
    target_opt_test['labels'] = target_test['labels']

    # get cost matrix
    Cs = source['covs']
    ys = source['labels']
    Ct_train = target_train['covs']
    Ct_test = target_test['covs']
    M = np.zeros((len(Cs), len(Ct_train)))
    for i, Cs_i in enumerate(Cs):
        for j, Ct_j in enumerate(Ct_train):
            M[i,j] = distance_riemann(Cs_i, Ct_j)**2

    # get the transportation plan
    mu_s = distribution_estimation_uniform(Cs)
    mu_t = distribution_estimation_uniform(Ct_train)
    gamma = sinkhorn_lpl1_mm(mu_s, ys, mu_t, M, reg=1.0)

    # transport the target matrices (train)
    Ct_train_transported = np.zeros(Ct_train.shape)
    for j in range(len(Ct_train_transported)):
        Ct_train_transported[j] = mean_riemann(Cs, sample_weight=gamma[:,j])
    target_opt_train['covs'] = Ct_train_transported

    # transport the target matrices (test)
    D = np.zeros((len(Ct_test), len(Ct_train)))
    for k, Ct_k in enumerate(Ct_test):
        for l, Ct_l in enumerate(Ct_train):
            D[k,l] = distance_riemann(Ct_k, Ct_l)**2
    idx = np.argmin(D, axis=1) # nearest neighbour to each target test matrix

    Ct_test_transported = np.zeros(Ct_test.shape)
    for i in range(len(Ct_test)):
        j = idx[i]

        Ci = Ct_test[i]
        Ri = Ct_train[j]
        Rf = Ct_train_transported[j]

        Ri_sqrt = sqrtm(Ri)
        Ri_invsqrt = invsqrtm(Ri)
        Li = logm(np.dot(Ri_invsqrt, np.dot(Ci, Ri_invsqrt)))
        eta_i = np.dot(Ri_sqrt, np.dot(Li, Ri_sqrt))

        Ri_Rf = geodesic_riemann(Rf, Ri, alpha=0.5)
        Ri_inv = np.linalg.inv(Ri)
        eta_f = np.dot(Ri_inv, np.dot(eta_i, Ri_inv))
        eta_f = np.dot(Ri_Rf, np.dot(eta_f, Ri_Rf))

        Rf_sqrt = sqrtm(Rf)
        Rf_invsqrt = invsqrtm(Rf)
        Ef = expm(np.dot(Rf_invsqrt, np.dot(eta_f, Rf_invsqrt)))
        Ct_test_transported[i] = np.dot(Rf_sqrt, np.dot(Ef, Rf_sqrt))

    target_opt_test['covs'] = Ct_test_transported

    return source, target_opt_train, target_opt_test

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

def transform_org2talmon_p300(source, target_train, target_test):

    covs_source = source['covs']
    covs_target_train = target_train['covs']
    covs_target_test = target_test['covs']

    weights = np.ones(len(source['labels']))
    weights[source['labels'] == 2] = 5
    M_source = mean_riemann(covs_source, sample_weight=weights)

    weights = np.ones(len(target_train['labels']))
    weights[target_train['labels'] == 2] = 5
    M_target_train = mean_riemann(covs_target_train, sample_weight=weights)

    M = geodesic_riemann(M_source, M_target_train, alpha=0.5)

    Gamma = ParallelTransport(reference_old=M_source, reference_new=M)
    covs_source_transp = Gamma.fit_transform(covs_source)

    Gamma = ParallelTransport(reference_old=M_target_train, reference_new=M)
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

def score_ensemble_org(settings, subject_target, ntop):

    dataset = settings['dataset']
    paradigm = settings['paradigm']
    session = settings['session']
    storage = settings['storage']
    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/TL_intra-subject_scores.pkl'
    acc_intra_dict = joblib.load(filepath)

    scores = []; subject_sources = [];
    for subject in settings['subject_list']:
        if subject == subject_target:
            continue
        else:
            scores.append(acc_intra_dict[subject])
            subject_sources.append(subject)
    scores = np.array(scores)

    subject_sources = np.array(subject_sources)
    idx_sort = scores.argsort()[::-1]
    scores = scores[idx_sort]
    subject_sources = subject_sources[idx_sort]
    subject_sources_ntop = subject_sources[:ntop]

    # get the geometric means for each subject (each class and also the center)
    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/subject_means.pkl'
    subj_means = joblib.load(filepath)

    # get the data for the target subject
    target_org = GD.get_dataset(dataset, subject_target, session, storage)
    if paradigm == 'MI':
        # things here are only implemented for MI for now
        target_org['covs'] = Covariances(estimator='oas').fit_transform(target_org['signals'])
        target_org['labels'] = target_org['labels']

    ncovs = settings['ncovs_list'][0]
    nrzt = 10
    score_rzt = 0.0
    for rzt in range(nrzt):

        # split randomly the target dataset
        target_org_train, target_org_test = get_target_split_motorimagery(target_org, ncovs)

        covs_train_target = target_org_train['covs']
        labs_train_target = target_org_train['labels']

        M1_target = mean_riemann(covs_train_target[labs_train_target == 'left_hand'])
        M2_target = mean_riemann(covs_train_target[labs_train_target == 'right_hand'])
        covs_train_target = np.stack([M1_target, M2_target])
        labs_train_target = np.array(['left_hand', 'right_hand'])

        clf = []
        for subj_source in subject_sources_ntop:

            M1_source = subj_means[subj_source]['left_hand']
            M2_source = subj_means[subj_source]['right_hand']
            covs_train_source = np.stack([M1_source, M2_source])
            labs_train_source = np.array(['left_hand', 'right_hand'])

            covs_train = np.concatenate([covs_train_source, covs_train_target])
            labs_train = np.concatenate([labs_train_source, labs_train_target])
            clfi = MDM()

            # problems here when using integer instead of floats on the sample_weight
            ntrials = 1.0*(target_org['covs'].shape[0])
            clfi.fit(covs_train, labs_train, sample_weight=np.array([ntrials/2, ntrials/2, 1.0*ncovs, 1.0*ncovs]))
            clf.append(clfi)

        covs_test = target_org_test['covs']
        labs_test = target_org_test['labels']

        ypred = []
        for clfi in clf:
            yi = clfi.predict(covs_test)
            ypred.append(yi)
        ypred = np.array(ypred)

        majorvoting = []
        for j in range(ypred.shape[1]):
            ypredj = ypred[:,j]
            values_unique, values_count = np.unique(ypredj, return_counts=True)
            majorvoting.append(values_unique[np.argmax(values_count)])
        majorvoting = np.array(majorvoting)

        score_rzt = score_rzt + np.mean(majorvoting == labs_test)

    score = score_rzt / nrzt

    return score

def score_pooling_org(settings, subject_target, ntop):

    dataset = settings['dataset']
    paradigm = settings['paradigm']
    session = settings['session']
    storage = settings['storage']

    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/TL_intra-subject_scores.pkl'
    acc_intra_dict = joblib.load(filepath)

    scores = []; subject_sources = [];
    for subject in settings['subject_list']:
        if subject == subject_target:
            continue
        else:
            scores.append(acc_intra_dict[subject])
            subject_sources.append(subject)
    scores = np.array(scores)

    subject_sources = np.array(subject_sources)
    idx_sort = scores.argsort()[::-1]
    scores = scores[idx_sort]
    subject_sources = subject_sources[idx_sort]
    subject_sources_ntop = subject_sources[:ntop]

    # get the geometric means for each subject (each class and also the center)
    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/subject_means.pkl'
    subj_means = joblib.load(filepath)

    # get the data for the target subject
    target_org = GD.get_dataset(dataset, subject_target, session, storage)
    if paradigm == 'MI':
        # things here are only implemented for MI for now
        target_org['covs'] = Covariances(estimator='oas').fit_transform(target_org['signals'])
        target_org['labels'] = target_org['labels']

    ncovs = settings['ncovs_list'][0]
    score_rzt = 0.0
    nrzt = 10
    for rzt in range(nrzt):

        # split randomly the target dataset
        target_org_train, target_org_test = get_target_split_motorimagery(target_org, ncovs)

        # get the data and pool it all together
        class_mean_1 = []
        class_mean_2 = []
        for subj_source in subject_sources_ntop:
            class_mean_1.append(subj_means[subj_source]['left_hand'])
            class_mean_2.append(subj_means[subj_source]['left_hand'])
        class_mean_1_source = np.stack(class_mean_1)
        class_mean_2_source = np.stack(class_mean_2)
        covs_train_source = np.concatenate([class_mean_1_source, class_mean_2_source])
        labs_train_source = np.concatenate([len(class_mean_1_source) * ['left_hand'], len(class_mean_2_source) * ['right_hand']])

        covs_train_target = target_org['covs']
        labs_train_target = target_org['labels']
        class_mean_1_target = mean_riemann(covs_train_target[labs_train_target == 'left_hand'])
        class_mean_2_target = mean_riemann(covs_train_target[labs_train_target == 'right_hand'])
        covs_train_target = np.stack([class_mean_1_target, class_mean_2_target])
        labs_train_target = np.array(['left_hand', 'right_hand'])

        covs_train = np.concatenate([covs_train_source, covs_train_target])
        labs_train = np.concatenate([labs_train_source, labs_train_target])

        covs_test = target_org_test['covs']
        labs_test = target_org_test['labels']

            # do the classification
        clf = MDM()
        clf.fit(covs_train, labs_train)
        score_rzt = score_rzt + clf.score(covs_test, labs_test)

    score = score_rzt / nrzt

    return score

def score_ensemble_rct(settings, subject_target, ntop):

    dataset = settings['dataset']
    paradigm = settings['paradigm']
    session = settings['session']
    storage = settings['storage']

    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/TL_intra-subject_scores.pkl'
    acc_intra_dict = joblib.load(filepath)

    scores = []; subject_sources = [];
    for subject in settings['subject_list']:
        if subject == subject_target:
            continue
        else:
            scores.append(acc_intra_dict[subject])
            subject_sources.append(subject)
    scores = np.array(scores)

    subject_sources = np.array(subject_sources)
    idx_sort = scores.argsort()[::-1]
    scores = scores[idx_sort]
    subject_sources = subject_sources[idx_sort]
    subject_sources_ntop = subject_sources[:ntop]

    # get the geometric means for each subject (each class and also the center)
    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/subject_means.pkl'
    subj_means = joblib.load(filepath)

    # get the data for the target subject
    target_org = GD.get_dataset(dataset, subject_target, session, storage)
    if paradigm == 'MI':
        # things here are only implemented for MI for now
        target_org['covs'] = Covariances(estimator='oas').fit_transform(target_org['signals'])
        target_org['labels'] = target_org['labels']

    ncovs = settings['ncovs_list'][0]
    nrzt = 10
    score_rzt = 0.0
    for rzt in range(nrzt):

        # split randomly the target dataset
        target_org_train, target_org_test = get_target_split_motorimagery(target_org, ncovs)

        covs_train_target = target_org_train['covs']
        labs_train_target = target_org_train['labels']

        MC_target = mean_riemann(covs_train_target)
        M1_target = mean_riemann(covs_train_target[labs_train_target == 'left_hand'])
        M2_target = mean_riemann(covs_train_target[labs_train_target == 'right_hand'])

        clf = []
        for subj_source in subject_sources_ntop:

            MC_source = subj_means[subj_source]['center']
            RefNew = np.eye(MC_target.shape[1])

            gamma_target = ParallelTransport(reference_old=MC_target, reference_new=RefNew)
            covs_train_target = np.stack([M1_target, M2_target])
            covs_train_target = gamma_target.fit_transform(covs_train_target)
            labs_train_target = np.array(['left_hand', 'right_hand'])

            gamma_source = ParallelTransport(reference_old=MC_source, reference_new=RefNew)
            M1_source = subj_means[subj_source]['left_hand']
            M2_source = subj_means[subj_source]['right_hand']
            covs_train_source = np.stack([M1_source, M2_source])
            covs_train_source = gamma_source.fit_transform(covs_train_source)
            labs_train_source = np.array(['left_hand', 'right_hand'])

            covs_train = np.concatenate([covs_train_source, covs_train_target])
            labs_train = np.concatenate([labs_train_source, labs_train_target])
            clfi = MDM()

            # problems here when using integer instead of floats on the sample_weight
            ntrials = 1.0*(target_org['covs'].shape[0])
            clfi.fit(covs_train, labs_train, sample_weight=np.array([ntrials/2, ntrials/2, 1.0*ncovs, 1.0*ncovs]))
            clf.append(clfi)

        covs_test = target_org_test['covs']
        covs_test = gamma_target.transform(covs_test)
        labs_test = target_org_test['labels']

        ypred = []
        for clfi in clf:
            yi = clfi.predict(covs_test)
            ypred.append(yi)
        ypred = np.array(ypred)

        majorvoting = []
        for j in range(ypred.shape[1]):
            ypredj = ypred[:,j]
            values_unique, values_count = np.unique(ypredj, return_counts=True)
            majorvoting.append(values_unique[np.argmax(values_count)])
        majorvoting = np.array(majorvoting)

        score_rzt = score_rzt + np.mean(majorvoting == labs_test)

    score = score_rzt / nrzt

    return score

def score_ensemble_prl(settings, subject_target, ntop):

    dataset = settings['dataset']
    paradigm = settings['paradigm']
    session = settings['session']
    storage = settings['storage']

    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/TL_intra-subject_scores.pkl'
    acc_intra_dict = joblib.load(filepath)

    scores = []; subject_sources = [];
    for subject in settings['subject_list']:
        if subject == subject_target:
            continue
        else:
            scores.append(acc_intra_dict[subject])
            subject_sources.append(subject)
    scores = np.array(scores)

    subject_sources = np.array(subject_sources)
    idx_sort = scores.argsort()[::-1]
    scores = scores[idx_sort]
    subject_sources = subject_sources[idx_sort]
    subject_sources_ntop = subject_sources[:ntop]

    # get the geometric means for each subject (each class and also the center)
    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/subject_means.pkl'
    subj_means = joblib.load(filepath)

    # get the data for the target subject
    target_org = GD.get_dataset(dataset, subject_target, session, storage)
    if paradigm == 'MI':
        # things here are only implemented for MI for now
        target_org['covs'] = Covariances(estimator='oas').fit_transform(target_org['signals'])
        target_org['labels'] = target_org['labels']

    ncovs = settings['ncovs_list'][0]
    nrzt = 10
    score_rzt = 0.0
    for rzt in range(nrzt):

        # split randomly the target dataset
        target_org_train, target_org_test = get_target_split_motorimagery(target_org, ncovs)

        covs_train_target = target_org_train['covs']
        labs_train_target = target_org_train['labels']

        MC_target = mean_riemann(covs_train_target)
        M1_target = mean_riemann(covs_train_target[labs_train_target == 'left_hand'])
        M2_target = mean_riemann(covs_train_target[labs_train_target == 'right_hand'])

        clf = []
        for subj_source in subject_sources_ntop:

            MC_source = subj_means[subj_source]['center']
            RefNew = geodesic_riemann(MC_source, MC_target, alpha=0.5)

            gamma_target = ParallelTransport(reference_old=MC_target, reference_new=RefNew)
            covs_train_target = np.stack([M1_target, M2_target])
            covs_train_target = gamma_target.fit_transform(covs_train_target)
            labs_train_target = np.array(['left_hand', 'right_hand'])

            gamma_source = ParallelTransport(reference_old=MC_source, reference_new=RefNew)
            M1_source = subj_means[subj_source]['left_hand']
            M2_source = subj_means[subj_source]['right_hand']
            covs_train_source = np.stack([M1_source, M2_source])
            covs_train_source = gamma_source.fit_transform(covs_train_source)
            labs_train_source = np.array(['left_hand', 'right_hand'])

            covs_train = np.concatenate([covs_train_source, covs_train_target])
            labs_train = np.concatenate([labs_train_source, labs_train_target])
            clfi = MDM()

            # problems here when using integer instead of floats on the sample_weight
            ntrials = 1.0*(target_org['covs'].shape[0])
            clfi.fit(covs_train, labs_train, sample_weight=np.array([ntrials/2, ntrials/2, 1.0*ncovs, 1.0*ncovs]))
            clf.append(clfi)

        covs_test = target_org_test['covs']
        covs_test = gamma_target.transform(covs_test)
        labs_test = target_org_test['labels']

        ypred = []
        for clfi in clf:
            yi = clfi.predict(covs_test)
            ypred.append(yi)
        ypred = np.array(ypred)

        majorvoting = []
        for j in range(ypred.shape[1]):
            ypredj = ypred[:,j]
            values_unique, values_count = np.unique(ypredj, return_counts=True)
            majorvoting.append(values_unique[np.argmax(values_count)])
        majorvoting = np.array(majorvoting)

        score_rzt = score_rzt + np.mean(majorvoting == labs_test)

    score = score_rzt / nrzt

    return score

def score_pooling_rct(settings, subject_target, ntop):

    dataset = settings['dataset']
    paradigm = settings['paradigm']
    session = settings['session']
    storage = settings['storage']

    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/TL_intra-subject_scores.pkl'
    acc_intra_dict = joblib.load(filepath)

    scores = []; subject_sources = [];
    for subject in settings['subject_list']:
        if subject == subject_target:
            continue
        else:
            scores.append(acc_intra_dict[subject])
            subject_sources.append(subject)
    scores = np.array(scores)

    subject_sources = np.array(subject_sources)
    idx_sort = scores.argsort()[::-1]
    scores = scores[idx_sort]
    subject_sources = subject_sources[idx_sort]
    subject_sources_ntop = subject_sources[:ntop]

    # get the geometric means for each subject (each class and also the center)
    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/subject_means.pkl'
    subj_means = joblib.load(filepath)

    # get the data for the target subject
    target_org = GD.get_dataset(dataset, subject_target, session, storage)
    if paradigm == 'MI':
        # things here are only implemented for MI for now
        target_org['covs'] = Covariances(estimator='oas').fit_transform(target_org['signals'])
        target_org['labels'] = target_org['labels']

    ncovs = settings['ncovs_list'][0]
    score_rzt = 0.0
    nrzt = 10
    for rzt in range(nrzt):

        # split randomly the target dataset
        target_org_train, target_org_test = get_target_split_motorimagery(target_org, ncovs)

        # get the data from the sources and pool it all together
        class_mean_1 = []
        class_mean_2 = []
        for subj_source in subject_sources_ntop:
            MC_source = subj_means[subj_source]['center']
            M1_source = subj_means[subj_source]['left_hand']
            M2_source = subj_means[subj_source]['right_hand']
            M1_source_rct = np.dot(invsqrtm(MC_source), np.dot(M1_source, invsqrtm(MC_source)))
            class_mean_1.append(M1_source_rct)
            M2_source_rct = np.dot(invsqrtm(MC_source), np.dot(M2_source, invsqrtm(MC_source)))
            class_mean_2.append(M2_source_rct)
        class_mean_1_source = np.stack(class_mean_1)
        class_mean_2_source = np.stack(class_mean_2)
        covs_train_source = np.concatenate([class_mean_1_source, class_mean_2_source])
        labs_train_source = np.concatenate([len(class_mean_1_source) * ['left_hand'], len(class_mean_2_source) * ['right_hand']])

        # re-center data for the target
        covs_train_target = target_org['covs']
        MC_target = mean_riemann(covs_train_target)
        labs_train_target = target_org['labels']
        class_mean_1_target = mean_riemann(covs_train_target[labs_train_target == 'left_hand'])
        class_mean_1_target = np.dot(invsqrtm(MC_target), np.dot(class_mean_1_target, invsqrtm(MC_target)))
        class_mean_2_target = mean_riemann(covs_train_target[labs_train_target == 'right_hand'])
        class_mean_2_target = np.dot(invsqrtm(MC_target), np.dot(class_mean_2_target, invsqrtm(MC_target)))

        covs_train_target = np.stack([class_mean_1_target, class_mean_2_target])
        labs_train_target = np.array(['left_hand', 'right_hand'])

        covs_train = np.concatenate([covs_train_source, covs_train_target])
        labs_train = np.concatenate([labs_train_source, labs_train_target])

        covs_test = target_org_test['covs']
        covs_test = np.stack([np.dot(invsqrtm(MC_target), np.dot(covi, invsqrtm(MC_target))) for covi in covs_test])
        labs_test = target_org_test['labels']

        # do the classification
        clf = MDM()
        clf.fit(covs_train, labs_train)
        score_rzt = score_rzt + clf.score(covs_test, labs_test)

    score = score_rzt / nrzt

    return score

def score_ensemble_rot(settings, subject_target, ntop):

    dataset = settings['dataset']
    paradigm = settings['paradigm']
    session = settings['session']
    storage = settings['storage']

    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/TL_intra-subject_scores.pkl'
    acc_intra_dict = joblib.load(filepath)

    scores = []; subject_sources = [];
    for subject in settings['subject_list']:
        if subject == subject_target:
            continue
        else:
            scores.append(acc_intra_dict[subject])
            subject_sources.append(subject)
    scores = np.array(scores)

    subject_sources = np.array(subject_sources)
    idx_sort = scores.argsort()[::-1]
    scores = scores[idx_sort]
    subject_sources = subject_sources[idx_sort]
    subject_sources_ntop = subject_sources[:ntop]

    # get the geometric means for each subject (each class and also the center)
    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/subject_means.pkl'
    subj_means = joblib.load(filepath)

    # get the data for the target subject
    target_org = GD.get_dataset(dataset, subject_target, session, storage)
    if paradigm == 'MI':
        # things here are only implemented for MI for now
        target_org['covs'] = Covariances(estimator='oas').fit_transform(target_org['signals'])
        target_org['labels'] = target_org['labels']

    ncovs = settings['ncovs_list'][0]
    nrzt = 10
    score_rzt = 0.0
    for rzt in range(nrzt):

        # split randomly the target dataset
        target_org_train, target_org_test = get_target_split_motorimagery(target_org, ncovs)

        covs_train_target = target_org_train['covs']
        labs_train_target = target_org_train['labels']

        MC_target = mean_riemann(covs_train_target)
        M1_target = mean_riemann(covs_train_target[labs_train_target == 'left_hand'])
        M2_target = mean_riemann(covs_train_target[labs_train_target == 'right_hand'])
        M1_target_rct = np.dot(invsqrtm(MC_target), np.dot(M1_target, invsqrtm(MC_target)))
        M2_target_rct = np.dot(invsqrtm(MC_target), np.dot(M2_target, invsqrtm(MC_target)))
        covs_train_target = np.stack([M1_target_rct, M2_target_rct])
        labs_train_target = np.array(['left_hand', 'right_hand'])

        clf = []
        for subj_source in subject_sources_ntop:

            MC_source = subj_means[subj_source]['center']
            M1_source = subj_means[subj_source]['left_hand']
            M2_source = subj_means[subj_source]['right_hand']
            M1_source_rct = np.dot(invsqrtm(MC_source), np.dot(M1_source, invsqrtm(MC_source)))
            M2_source_rct = np.dot(invsqrtm(MC_source), np.dot(M2_source, invsqrtm(MC_source)))

            M = [M1_target_rct, M2_target_rct]
            Mtilde = [M1_source_rct, M2_source_rct]
            R = manifoptim.get_rotation_matrix(M, Mtilde)
            M1_source_rot = np.dot(R, np.dot(M1_source_rct, R.T))
            M2_source_rot = np.dot(R, np.dot(M2_source_rct, R.T))

            covs_train_source = np.stack([M1_source_rot, M2_source_rot])
            labs_train_source = np.array(['left_hand', 'right_hand'])

            covs_train = np.concatenate([covs_train_source, covs_train_target])
            labs_train = np.concatenate([labs_train_source, labs_train_target])
            clfi = MDM()

            # problems here when using integer instead of floats on the sample_weight
            ntrials = 1.0*(target_org['covs'].shape[0])
            clfi.fit(covs_train, labs_train, sample_weight=np.array([ntrials/2, ntrials/2, 1.0*ncovs, 1.0*ncovs]))
            clf.append(clfi)

        covs_test = target_org_test['covs']
        covs_test = np.stack([invsqrtm(MC_target) @ covi @ invsqrtm(MC_target) for covi in covs_test])
        labs_test = target_org_test['labels']

        ypred = []
        for clfi in clf:
            yi = clfi.predict(covs_test)
            ypred.append(yi)
        ypred = np.array(ypred)

        majorvoting = []
        for j in range(ypred.shape[1]):
            ypredj = ypred[:,j]
            values_unique, values_count = np.unique(ypredj, return_counts=True)
            majorvoting.append(values_unique[np.argmax(values_count)])
        majorvoting = np.array(majorvoting)

        score_rzt = score_rzt + np.mean(majorvoting == labs_test)

    score = score_rzt / nrzt

    return score

def score_pooling_rot(settings, subject_target, ntop):

    dataset = settings['dataset']
    paradigm = settings['paradigm']
    session = settings['session']
    storage = settings['storage']

    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/TL_intra-subject_scores.pkl'
    acc_intra_dict = joblib.load(filepath)

    scores = []; subject_sources = [];
    for subject in settings['subject_list']:
        if subject == subject_target:
            continue
        else:
            scores.append(acc_intra_dict[subject])
            subject_sources.append(subject)
    scores = np.array(scores)

    subject_sources = np.array(subject_sources)
    idx_sort = scores.argsort()[::-1]
    scores = scores[idx_sort]
    subject_sources = subject_sources[idx_sort]
    subject_sources_ntop = subject_sources[:ntop]

    # get the geometric means for each subject (each class and also the center)
    filepath = '/nethome/coelhorp/Development/python/riemann-lab-stable/riemann_lab/helpers/transfer_learning/auxiliarydata/' + dataset + '/subject_means.pkl'
    subj_means = joblib.load(filepath)

    # get the data for the target subject
    target_org = GD.get_dataset(dataset, subject_target, session, storage)
    if paradigm == 'MI':
        # things here are only implemented for MI for now
        target_org['covs'] = Covariances(estimator='oas').fit_transform(target_org['signals'])
        target_org['labels'] = target_org['labels']

    ncovs = settings['ncovs_list'][0]
    score_rzt = 0.0
    nrzt = 10
    for rzt in range(nrzt):

        # split randomly the target dataset
        target_org_train, target_org_test = get_target_split_motorimagery(target_org, ncovs)

        # get the data from the sources and pool it all together
        class_mean_1 = []
        class_mean_2 = []
        for subj_source in subject_sources_ntop:
            MC_source = subj_means[subj_source]['center']
            M1_source = subj_means[subj_source]['left_hand']
            M2_source = subj_means[subj_source]['right_hand']
            M1_source_rct = np.dot(invsqrtm(MC_source), np.dot(M1_source, invsqrtm(MC_source)))
            class_mean_1.append(M1_source_rct)
            M2_source_rct = np.dot(invsqrtm(MC_source), np.dot(M2_source, invsqrtm(MC_source)))
            class_mean_2.append(M2_source_rct)
        class_mean_1_source = np.stack(class_mean_1)
        class_mean_2_source = np.stack(class_mean_2)
        covs_train_source = np.concatenate([class_mean_1_source, class_mean_2_source])
        labs_train_source = np.concatenate([len(class_mean_1_source) * ['left_hand'], len(class_mean_2_source) * ['right_hand']])

        # re-center data for the target
        covs_train_target = target_org['covs']
        MC_target = mean_riemann(covs_train_target)
        labs_train_target = target_org['labels']
        class_mean_1_target = mean_riemann(covs_train_target[labs_train_target == 'left_hand'])
        class_mean_1_target_rct = np.dot(invsqrtm(MC_target), np.dot(class_mean_1_target, invsqrtm(MC_target)))
        class_mean_2_target = mean_riemann(covs_train_target[labs_train_target == 'right_hand'])
        class_mean_2_target_rct = np.dot(invsqrtm(MC_target), np.dot(class_mean_2_target, invsqrtm(MC_target)))

        # rotate the source matrices for each subject with respect to the target
        class_mean_1_source_rot = []
        class_mean_2_source_rot = []
        for Mi1, Mi2 in zip(class_mean_1_source, class_mean_2_source):
            M = [class_mean_1_target_rct, class_mean_2_target_rct]
            Mtilde = [Mi1, Mi2]
            U = manifoptim.get_rotation_matrix(M, Mtilde)
            Mi1_rot = np.dot(U, np.dot(Mi1, U.T))
            class_mean_1_source_rot.append(Mi1_rot)
            Mi2_rot = np.dot(U, np.dot(Mi2, U.T))
            class_mean_2_source_rot.append(Mi2_rot)

        class_mean_1_source = np.stack(class_mean_1_source_rot)
        class_mean_2_source = np.stack(class_mean_2_source_rot)
        covs_train_source = np.concatenate([class_mean_1_source, class_mean_2_source])

        covs_train_target = np.stack([class_mean_1_target_rct, class_mean_2_target_rct])
        labs_train_target = np.array(['left_hand', 'right_hand'])

        covs_train = np.concatenate([covs_train_source, covs_train_target])
        labs_train = np.concatenate([labs_train_source, labs_train_target])

        covs_test = target_org_test['covs']
        covs_test = np.stack([np.dot(invsqrtm(MC_target), np.dot(covi, invsqrtm(MC_target))) for covi in covs_test])
        labs_test = target_org_test['labels']

        # do the classification
        clf = MDM()
        clf.fit(covs_train, labs_train)
        score_rzt = score_rzt + clf.score(covs_test, labs_test)

    score = score_rzt / nrzt

    return score
