"""
We would like to thank Xiaolu Jiang for her contribution to data preprocessing.
Author : Xiaolu Jiang
modified by Wei Zhao

Citation
Hope this code can be useful. I would appreciate you citing us in your paper. 😊

Zhao, W., Jiang, X., Zhang, B. et al. CTNet: a convolutional transformer network for EEG-based motor imagery classification. Sci Rep 14, 20237 (2024). https://doi.org/10.1038/s41598-024-71118-7


"""

import mne
import numpy as np
import scipy.signal as signal
from scipy.io import savemat
import scipy.io as sio
import numpy as np

# 读取训练集和对应的标签到mat文件
for nSub in range(1, 10):
    data_sub = np.empty((0, 3, 1000)) 
    labels_sub = np.empty((0, 1))  
    for nSes in range(1,4):
        # Load the gdf file
        raw = mne.io.read_raw_gdf('./BCICIV_2b_gdf/'+'B0%d0%dT.gdf' % (nSub,nSes))  
        # Select the events of interest
        # Events is the data at each time point, and event_dict is the correspondence between the label and the label sequence number.
        events, event_dict = mne.events_from_annotations(raw) 
        event_id = {'Left': event_dict['769'], 'Right': event_dict['770']}
        # Select the events corresponding to the four categories we are concerned about. Here events[:, 2] refers to the third column in events, that is, the event number.
        selected_events = events[np.isin(events[:, 2], list(event_id.values()))]  
        
        # Select the removed channel, that is, the EOG channel does not participate in classification
        raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                        exclude='bads')
        # Epoch the data
        epochs = mne.Epochs(raw, selected_events, event_id, picks=picks,tmin=0,tmax=3.996,preload=True,baseline=None)


        # Get the labels
        #labels = epochs.events[:, 2]
        mat = sio.loadmat('./true_labels/'+'B0%d0%dT.mat'% (nSub,nSes)) 
        labels = mat['classlabel']

        # Store the data and labels of each epoch into data_sub and labels_sub
        data_sub = np.vstack((data_sub, epochs.get_data()))
        labels_sub = np.vstack((labels_sub, labels))
    
    # Output the shape of data_sub and labels_sub to ensure that it is consistent with the data in the mat file required by conformer.py
    print('B0%dT:'%nSub,data_sub.shape, labels_sub.shape)    
    # Save the data and labels to a .mat file
    savemat('./mymat_withoutFilter/B0%dT.mat' % nSub, {'data': data_sub, 'label': labels_sub})


# Read the test set and the corresponding labels into a mat file
for nSub in range(1, 10):
    data_sub = np.empty((0, 3, 1000)) 
    labels_sub = np.empty((0, 1)) 
    for nSes in range(4,6):
        # Load the gdf file
        raw = mne.io.read_raw_gdf('./BCICIV_2b_gdf/'+'B0%d0%dE.gdf' % (nSub,nSes)) 

        # Select the events of interest
        events, event_dict = mne.events_from_annotations(raw)
        event_id = {'Unknown': event_dict['783']}  
        selected_events = events[np.isin(events[:, 2], list(event_id.values()))]

        raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']    
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                        exclude='bads') 
        
        # Epoch the data
        epochs = mne.Epochs(raw, selected_events,event_id,picks=picks,tmin=0,tmax=3.996,preload=True,baseline=None,on_missing='ignore')
    
        # Get the labels
        mat = sio.loadmat('./true_labels/'+'B0%d0%dE.mat'% (nSub,nSes))
        labels = mat['classlabel']  

        data_sub = np.vstack((data_sub, epochs.get_data()))
        labels_sub = np.vstack((labels_sub, labels))
    
    print('B0%dE:'%nSub,data_sub.shape, labels_sub.shape)
    # Save the data and labels to a .mat file
    savemat('./mymat_withoutFilter/B0%dE.mat' % nSub, {'data': data_sub, 'label': labels_sub})  
