"""
# Function: Read files and preprocess them
# Steps:
# 1. Import data from the gdf file provided before the competition, 
#    remove unwanted channels, and select required events.
# 2. Select desired time segments for slicing; treat each segment (4s) as one sample.
# 3. Import labels from the mat file provided after the competition, 
#    ensuring they correspond with epochs and their numbers match.
# 4. Save the resulting data in a new mat file, 
#    preparing it for use in the subsequent main.py.
"""

import mne
import numpy as np
import scipy.signal as signal
from scipy.io import savemat
import scipy.io as sio
import numpy as np

def changeGdf2Mat(dir_path, mode="train"):
    '''
    read data from GDF files and store as mat files

    Parameters
    ----------
    dir_path : str
        GDF file dir path.
    mode : str, optional
        change train dataset or eval dataset. The default is "train".

    Returns
    -------
    None.

    '''
    mode_str = ''
    if mode=="train":
        mode_str = 'T'
    else:
        mode_str = 'E'
    for nSub in range(1, 10):
        # Load the gdf file
        data_filename = dir_path+'BCICIV_2a_gdf/A0{}{}.gdf'.format(nSub, mode_str)
        raw = mne.io.read_raw_gdf(data_filename)  
    
        # Select the events of interest
        events, event_dict = mne.events_from_annotations(raw) 
        if mode=="train":
            # train dataset are labeled
            event_id = {'Left': event_dict['769'],
                        'Right': event_dict['770'], 
                        'Foot': event_dict['771'],
                        'Tongue': event_dict['772']}  
        else:
            # evaluate dataset are labeled as 'Unknnow'
            event_id = {'Unknown': event_dict['783']}
            
        # Select the events corresponding to the four categories we are interested in. Here, events[:, 2] refers to the third column of the events array, which represents the event IDs.
        selected_events = events[np.isin(events[:, 2], list(event_id.values()))]  
        
        # remove EOG channels
        raw.info['bads'] += ['EOG-left', 'EOG-central', 'EOG-right']
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False, exclude='bads')
        # Epoch the data
        # using 4s (1000 sample point ) segmentation
        epochs = mne.Epochs(raw, selected_events, event_id, picks=picks,tmin=0, tmax=3.996, preload=True, baseline=None)
        
        filtered_data = epochs.get_data()
        label_filename = dir_path + 'true_labels/'+'A0{}{}.mat'.format(nSub, mode_str)
        mat = sio.loadmat(label_filename)  # load target mat file
        labels = mat['classlabel']
       
        # Save the data and labels to a .mat file
        result_filename = 'mymat_raw/A0{}{}.mat'.format(nSub, mode_str)
        savemat(result_filename, {'data': filtered_data, 'label': labels})

dir_path = './'
# prepare train dataset
changeGdf2Mat(dir_path, 'train')
# prepare test dataset
changeGdf2Mat(dir_path, 'eval')

