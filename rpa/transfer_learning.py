#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:57:12 2018

@author: coelhorp
"""

import numpy as np
from sklearn.metrics import roc_auc_score

from .helpers.transfer_learning.utils import transform_org2rct, transform_rct2str, transform_rct2rot, transform_org2opt, transform_org2talmon
from .helpers.transfer_learning.utils import transform_org2rct_p300, transform_org2talmon_p300, transform_rct2rot_p300
from .helpers.transfer_learning.utils import get_sourcetarget_split_motorimagery, get_sourcetarget_split_p300

def RPA_recenter(source, target_train, target_test, paradigm='MI', weight_samples=False):
    if paradigm == 'P300':
        return transform_org2rct_p300(source, target_train, target_test, weight_samples)
    else:
        return transform_org2rct(source, target_train, target_test)

def RPA_stretch(source, target_train, target_test, paradigm='MI'):
    return transform_rct2str(source, target_train, target_test)

def RPA_rotate(source, target_train, target_test, paradigm='MI', class_weights=None, distance='euc'):
    if paradigm == 'P300':
        return transform_rct2rot_p300(source, target_train, target_test, class_weights, distance)
    else:
        return transform_rct2rot(source, target_train, target_test, class_weights, distance)

def RPA_correct_driftruns(data_org, runs_dict):

    # runs_dict is a dictionary
    # runs_dict[i] = idx_run_i

    covs_org = data_org['covs']
    labs_org = data_org['labels']

    covs_run = {}; labs_run = {}
    for runi in runs_dict.keys():
        covs_run[runi] = covs_org[runs_dict[runi]]
        labs_run[runi] = labs_org[runs_dict[runi]]

    covs_run_rct = {}
    for runi in runs_dict.keys():
        target_org_train = {}
        target_org_train['covs'] = covs_run[runi]
        target_org_train['labels'] = labs_run[runi]
        target_org_test = target_org_train
        _, target_rct_train, _ = RPA_recenter(target_org_train, target_org_train, target_org_test)
        covs_run_rct[runi] = target_rct_train['covs']

    data_rct = {}
    data_rct['covs'] = np.concatenate([covs_run_rct[runi] for runi in runs_dict.keys()], axis=0)
    data_rct['labels'] = np.concatenate([labs_run[runi] for runi in runs_dict.keys()], axis=0)

    return data_rct

def OptimTransp(source, target_train, target_test, paradigm='MI'):
    return transform_org2opt(source, target_train, target_test)

def TalmonTransport(source, target_train, target_test, paradigm='MI'):
    if paradigm == 'P300':
        return transform_org2talmon_p300(source, target_train, target_test)
    else:
        return transform_org2talmon(source, target_train, target_test)

def get_sourcetarget_split(source, target, ncovs_train, paradigm='MI'):
    if (paradigm == 'P300'):
        return get_sourcetarget_split_p300(source, target, ncovs_train)
    else:
        return get_sourcetarget_split_motorimagery(source, target, ncovs_train)

def get_score_notransfer(clf, target_train, target_test, paradigm='MI'):

    covs_train = target_train['covs']
    y_train = target_train['labels']
    covs_test = target_test['covs']
    y_test = target_test['labels']

    clf.fit(covs_train, y_train)

    y_pred = clf.predict(covs_test)

    y_test = np.array([y_test == i for i in np.unique(y_test)]).T
    y_pred = np.array([y_pred == i for i in np.unique(y_pred)]).T

    return roc_auc_score(y_test, y_pred)

def get_score_transferlearning(clf, source, target_train, target_test, paradigm='MI'):

    covs_source, y_source = source['covs'], source['labels']
    covs_target_train, y_target_train = target_train['covs'], target_train['labels']
    covs_target_test, y_target_test = target_test['covs'], target_test['labels']

    covs_train = np.concatenate([covs_source, covs_target_train])
    y_train = np.concatenate([y_source, y_target_train])
    clf.fit(covs_train, y_train)

    covs_test = covs_target_test
    y_test = y_target_test

    y_pred = clf.predict(covs_test)

    y_test = np.array([y_test == i for i in np.unique(y_test)]).T
    y_pred = np.array([y_pred == i for i in np.unique(y_pred)]).T

    return roc_auc_score(y_test, y_pred)

