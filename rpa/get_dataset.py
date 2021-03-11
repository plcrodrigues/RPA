import joblib
import os

# load the source and target datasets
def get_dataset(datafolder, paradigm, subject=1):
    
    datapath = {}
    datapath['motorimagery'] = os.path.join(datafolder, 'MOTOR-IMAGERY') 
    datapath['ssvep'] = os.path.join(datafolder, 'SSVEP')
    
    filepath = os.path.join(datapath[paradigm],'subject_' + str(subject).zfill(2) + '.pkl')
    data = joblib.load(filepath)
        
    return data