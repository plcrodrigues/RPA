#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:33:59 2017

@author: coelhorp
"""

from collections import OrderedDict
import numpy as np

from scipy.linalg import eigh

from pyriemann.estimation import Covariances
from pyriemann.utils.mean import mean_riemann, geodesic_riemann
from pyriemann.utils.base import invsqrtm, sqrtm, logm, expm, powm
from pyriemann.utils.distance import distance_riemann
from pyriemann.clustering import Potato   

from ot.da import distribution_estimation_uniform    
from ot.da import sinkhorn_lpl1_mm 

from sklearn.externals import joblib
import transferlearning_manopt as manifoptim

def get_dataset(paradigm='motorimagery', subject=1):
    
    datapath = {}
    datapath['motorimagery'] = './datasets/MOTOR-IMAGERY/' 
    datapath['ssvep'] = './datasets/SSVEP/'
    
    filepath = datapath[paradigm] + 'subject_' + str(subject).zfill(2) + '.pkl'
    data = joblib.load(filepath)
    
    # data.keys() = ['covs', 'labs']
    return data

def get_target_split(target, ncovs_train):

    target_train_idx = []
    for j in np.unique(target['labs']):
        sel = np.arange(np.sum(target['labs'] == j))                
        np.random.shuffle(sel)
        target_train_idx.append(np.arange(len(target['labs']))[target['labs'] == j][sel[:ncovs_train]])
    target_train_idx = np.concatenate(target_train_idx)
    target_test_idx  = np.array([i for i in range(len(target['labs'])) if i not in target_train_idx])

    target_train = {}
    target_train['covs'] = target['covs'][target_train_idx]
    target_train['labs'] = target['labs'][target_train_idx]

    target_test = {}
    target_test['covs'] = target['covs'][target_test_idx]
    target_test['labs'] = target['labs'][target_test_idx]

    return target_train, target_test

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
    source_rct['labs'] = source_org['labs']

    M_target = mean_riemann(target_org_train['covs'])

    target_rct_train = {}     
    T_target_train = np.stack([M_target]*len(target_org_train['covs']))       
    target_rct_train['covs'] = parallel_transport_covariances(target_org_train['covs'], T_target_train) 
    target_rct_train['labs'] = target_org_train['labs']    

    target_rct_test = {}    
    T_target_test = np.stack([M_target]*len(target_org_test['covs']))         
    target_rct_test['covs'] = parallel_transport_covariances(target_org_test['covs'], T_target_test) 
    target_rct_test['labs'] = target_org_test['labs']    
    
    return source_rct, target_rct_train, target_rct_test  

def transform_rct2str(source_rct, target_rct_train, target_rct_test):
    '''

    '''      

    covs_source = source_rct['covs']    
    covs_target_train = target_rct_train['covs'] 
    covs_target_test = target_rct_test['covs'] 
    
    source_str  = {}
    source_str ['covs'] = source_rct['covs']
    source_str ['labs'] = source_rct['labs']

    n = covs_source.shape[1]    
    disp_source = np.sum([distance_riemann(covi, np.eye(n)) ** 2 for covi in covs_source]) / len(covs_source)
    disp_target = np.sum([distance_riemann(covi, np.eye(n)) ** 2 for covi in covs_target_train]) / len(covs_target_train)
    p = np.sqrt(disp_target / disp_source)
    
    target_str_train  = {}
    target_str_train ['covs'] = np.stack([powm(covi, 1.0/p) for covi in covs_target_train])
    target_str_train ['labs'] = target_rct_train['labs']
    
    target_str_test  = {}
    target_str_test ['covs'] = np.stack([powm(covi, 1.0/p) for covi in covs_target_test])
    target_str_test ['labs'] = target_rct_test['labs']    

    return source_str , target_str_train, target_str_test

def transform_str2rot(source_str, target_str_train, target_str_test):
    '''
        rotate the re-centered matrices from the target so they match with those from source
        note that we use information from class labels of some of the target covariances (not all but some)
        rct (re-centered matrices) -> rot (rotated re-centered matrices)
    '''         

    source_rot = {}
    source_rot['covs'] = source_str['covs']
    source_rot['labs'] = source_str['labs']
    
    target_rot_train = {}
    target_rot_train['labs'] = target_str_train['labs']
    
    target_rot_test = {}
    target_rot_test['labs'] = target_str_test['labs']

    M_source = []
    for i in np.unique(source_str['labs']):
        M_source_i = mean_riemann(source_str['covs'][source_str['labs'] == i])
        M_source.append(M_source_i)

    M_target_train = []
    for j in np.unique(target_str_train['labs']):
        M_target_train_j = mean_riemann(target_str_train['covs'][target_str_train['labs'] == j])            
        M_target_train.append(M_target_train_j)  
            
    R = manifoptim.get_rotation_matrix(M=M_source, Mtilde=M_target_train)
    
    covs_target_train = np.stack([np.dot(R, np.dot(covi, R.T)) for covi in target_str_train['covs']])
    target_rot_train['covs'] = covs_target_train

    covs_target_test = np.stack([np.dot(R, np.dot(covi, R.T)) for covi in target_str_test['covs']])
    target_rot_test['covs'] = covs_target_test

    return source_rot, target_rot_train, target_rot_test

def get_score_transferlearning(clf, source, target_train, target_test):  
    
    covs_source, y_source = source['covs'], source['labs']
    covs_target_train, y_target_train = target_train['covs'], target_train['labs']
    covs_target_test, y_target_test = target_test['covs'], target_test['labs']        

    covs_train = np.concatenate([covs_source, covs_target_train])
    y_train = np.concatenate([y_source, y_target_train])
    clf.fit(covs_train, y_train)

    covs_test = covs_target_test
    y_test = y_target_test
    scr = clf.score(covs_test, y_test)    

    return scr    

def get_scores_calibration(clf, target_train, target_test):        
    
    covs_train = target_train['covs']
    y_train = target_train['labs']
    covs_test = target_test['covs']
    y_test = target_test['labs']
    
    clf.fit(covs_train, y_train)
    return clf.score(covs_test, y_test)







