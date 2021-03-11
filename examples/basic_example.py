
# installed via pip
import numpy as np
import matplotlib.pyplot as plt
import joblib
from pyriemann.classification import MDM
from pyriemann.utils.distance import distance_riemann 
from tqdm import tqdm
from collections import OrderedDict

# get the functions from RPA package
from rpa import transfer_learning as TL
from rpa import diffusion_map as DM
from rpa import get_dataset as GD

# get the dataset
datafolder = '../datasets/'
paradigm = 'motorimagery'
target = GD.get_dataset(datafolder, paradigm, subject=3) 
source = GD.get_dataset(datafolder, paradigm, subject=1)

# instantiate the Riemannian classifier to use
clf = MDM() 

# create a scores dictionary
methods_list = ['org', 'rct', 'rpa', 'clb']
scores = OrderedDict()
for method in methods_list:
    scores[method] = []

nrzt = 5
for _ in tqdm(range(nrzt)):

    # get the split for the source and target dataset
    source_org, target_org_train, target_org_test = TL.get_sourcetarget_split(source, target, ncovs_train=10)

    # get the score with the original dataset
    scores['org'].append(TL.get_score_transferlearning(clf, source_org, target_org_train, target_org_test))

    # get the score with the re-centered matrices
    source_rct, target_rct_train, target_rct_test = TL.RPA_recenter(source_org, target_org_train, target_org_test)
    scores['rct'].append(TL.get_score_transferlearning(clf, source_rct, target_rct_train, target_rct_test))

    # rotate the re-centered-stretched matrices using information from classes
    source_rpa, target_rpa_train, target_rpa_test = TL.RPA_rotate(source_rct, target_rct_train, target_rct_test)
    scores['rpa'].append(TL.get_score_transferlearning(clf, source_rpa, target_rpa_train, target_rpa_test))

    # get the score without any transformation
    scores['clb'].append(TL.get_score_notransfer(clf, target_org_train, target_org_test))
    
for method in methods_list:
    scores[method] = np.mean(scores[method])

for meth in ['org', 'rct', 'rpa', 'clb']:
    print(meth, '{0:.2f}'.format(scores[meth]))   