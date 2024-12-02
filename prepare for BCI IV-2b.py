
import mne
import numpy as np
import scipy.signal as signal
from scipy.io import savemat
import scipy.io as sio
import numpy as np

# 读取训练集和对应的标签到mat文件
for nSub in range(1, 10):
    # 新建两个空的ndarray，用于存放数据和标签
    data_sub = np.empty((0, 3, 1000))  # 3个通道，每个通道1000个样本点
    labels_sub = np.empty((0, 1))  # 1个标签，即左手还是右手
    for nSes in range(1,4):
        # Load the gdf file
        raw = mne.io.read_raw_gdf('./BCICIV_2b_gdf/'+'B0%d0%dT.gdf' % (nSub,nSes))  # 比赛开始前提供的gdf数据

        # Select the events of interest
        events, event_dict = mne.events_from_annotations(raw) # events是每个时间点的数据，event_dict是标签与标签序号的对应
        event_id = {'Left': event_dict['769'], 'Right': event_dict['770']}
        selected_events = events[np.isin(events[:, 2], list(event_id.values()))]  # 选取我们关心的四个类别对应的事件，这里events[:, 2]是指events中的第三列，即事件的编号。
        
        # 选择去掉的channel，即EOG通道不参与分类
        raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                        exclude='bads')
        # Epoch the data
        # 一段（epoch）时间的样本点作为一个input，会对应一个target
        # 1. 在raw中将EOG通道去掉
        # 2. 截取的片段为时间t=2s到t=6s，对应到event里是0s到4s，为保证数据是1000个样本，不取4，取3.996
        epochs = mne.Epochs(raw, selected_events, event_id, picks=picks,tmin=0,tmax=3.996,preload=True,baseline=None)

        # Apply the Chebyshev filter
        # 去除50hz数据，滤通通道为4hz~40hz之间
        # sos = signal.cheby2(6, 50, [4, 40], btype='band', output='sos', fs=raw.info['sfreq'])
        # filtered_data = signal.sosfilt(sos, epochs.get_data())
        filtered_data = epochs.get_data()

        # Get the labels
        #labels = epochs.events[:, 2]
        mat = sio.loadmat('./true_labels/'+'B0%d0%dT.mat'% (nSub,nSes))  # 竞赛后官方公布的target，是mat文件
        labels = mat['classlabel']
        # print(len(filtered_data), len(labels)) # 确保输出的数据每个维度和conformer.py要求输入的mat文件里的数据一致
        #print('B0%d0%dT:'%(nSub,nSes),filtered_data.shape, labels.shape)

        # 将每个epoch的数据和标签存入data_sub和labels_sub
        data_sub = np.vstack((data_sub, filtered_data))
        labels_sub = np.vstack((labels_sub, labels))
    
    # 输出data_sub和labels_sub的shape，确保和conformer.py要求输入的mat文件里的数据一致
    print('B0%dT:'%nSub,data_sub.shape, labels_sub.shape)    
    # Save the data and labels to a .mat file
    savemat('./mymat_raw/B0%dT.mat' % nSub, {'data': data_sub, 'label': labels_sub})


# 读取测试集和对应的标签到mat文件
for nSub in range(1, 10):
    data_sub = np.empty((0, 3, 1000))  # 3个通道，每个通道1000个样本点
    labels_sub = np.empty((0, 1))  # 1个标签，即左手还是右手
    for nSes in range(4,6):
        # Load the gdf file
        raw = mne.io.read_raw_gdf('./BCICIV_2b_gdf/'+'B0%d0%dE.gdf' % (nSub,nSes)) 

        # Select the events of interest
        events, event_dict = mne.events_from_annotations(raw)
        event_id = {'Unknown': event_dict['783']}  # 测试集的样本一开始以unknown作为标签，选择unknown标签的数据来切片
        selected_events = events[np.isin(events[:, 2], list(event_id.values()))]

        raw.info['bads'] += ['EOG:ch01', 'EOG:ch02', 'EOG:ch03']    
        picks = mne.pick_types(raw.info, meg=False, eeg=True, eog=False, stim=False,
                        exclude='bads') 
        
        # Epoch the data
        epochs = mne.Epochs(raw, selected_events,event_id,picks=picks,tmin=0,tmax=3.996,preload=True,baseline=None,on_missing='ignore')
    
        # Get the labels
        mat = sio.loadmat('./true_labels/'+'B0%d0%dE.mat'% (nSub,nSes))
        labels = mat['classlabel']  

        # Apply the Chebyshev filter
        sos = signal.cheby2(6, 50, [4, 40], btype='band', output='sos', fs=raw.info['sfreq'])
        filtered_data = signal.sosfilt(sos, epochs.get_data())  # filtered_data.shape=(288, 22, 1000)=（epoch个数，通道数，样本点数）

        # print('B0%d0%dE:'%(nSub,nSes),filtered_data.shape, labels.shape)  # 确保输出的数据每个维度和conformer.py要求输入的mat文件里的数据一致
        # print(len(reshaped_data), len(labels))

        # 将每个epoch的数据和标签存入data_sub和labels_sub
        data_sub = np.vstack((data_sub, filtered_data))
        labels_sub = np.vstack((labels_sub, labels))
    
    # 输出data_sub和labels_sub的shape，确保和conformer.py要求输入的mat文件里的数据一致
    print('B0%dE:'%nSub,data_sub.shape, labels_sub.shape)
    # Save the data and labels to a .mat file
    savemat('./mymat_raw/B0%dE.mat' % nSub, {'data': data_sub, 'label': labels_sub})  
