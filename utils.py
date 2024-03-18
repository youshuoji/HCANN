import mne
import mne.io
import numpy as np
from scipy.stats import zscore
# from .mapping import ch_names, ch_types
import os
import sys
sys.path.append(os.getcwd())
import random

def make_dataset(load_path, t_min=0.5, t_max=1.5,
                 sfreq=256, t_ratio=0.7, v_ratio=0.2,
                 standardization=True,
                 category_type='fine'):
    datas = np.load(load_path)['events_data']
    events_data = []

    for i in range(len(datas)):
        events_data.append(datas[i][:, :, int((t_min)*sfreq):int((t_max)*sfreq)])
        if standardization:
            for trial in range(events_data[i].shape[0]):
                for chan in range(events_data[i].shape[1]):
                    events_data[i][trial, chan, :] = zscore(events_data[i][trial, chan, :])
    index = [p for p in range(events_data[0].shape[0])]

    for i in range(8):
        random.shuffle(index)
        events_data[i] = events_data[i][index, :, :]

    train_X = []
    train_Y = []
    valid_X = []
    valid_Y = []
    test_X = []
    test_Y = []

    if category_type == 'fine':
        for i in range(len(events_data)):
            train_X.append(events_data[i][:int(t_ratio * events_data[i].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[i].shape[0]))))
            test_X.append(events_data[i][int((t_ratio) * events_data[i].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio) * events_data[i].shape[0]))))
            # valid_X.append(events_data[i][int(t_ratio * events_data[i].shape[0]):int((t_ratio + v_ratio) * events_data[i].shape[0])])
            # valid_Y.append(i * np.ones((int(v_ratio * events_data[i].shape[0]), 1)))
            # test_X.append(events_data[i][int((t_ratio + v_ratio) * events_data[i].shape[0]):])
            # test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[i].shape[0]), 1)))
    elif category_type == 'coarse':
        for i in range(len(events_data)):
            train_X.append(events_data[i][:int(t_ratio * events_data[i].shape[0])])
            train_Y.append((i > 3) * np.ones((int(t_ratio * events_data[i].shape[0]), 1)))
            valid_X.append(events_data[i][int(t_ratio * events_data[i].shape[0]):int((t_ratio + v_ratio) * events_data[i].shape[0])])
            valid_Y.append((i > 3) * np.ones((int(v_ratio * events_data[i].shape[0]), 1)))
            test_X.append(events_data[i][int((t_ratio + v_ratio) * events_data[i].shape[0]):])
            test_Y.append((i > 3) * np.ones((int((1 - t_ratio - v_ratio) * events_data[i].shape[0]), 1)))
    elif category_type == 'fruit':
        for i in range(4):
            train_X.append(events_data[i][:int(t_ratio * events_data[i].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[i].shape[0]), 1)))
            valid_X.append(events_data[i][int(t_ratio * events_data[i].shape[0]):int((t_ratio + v_ratio) * events_data[i].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[i].shape[0]), 1)))
            test_X.append(events_data[i][int((t_ratio + v_ratio) * events_data[i].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[i].shape[0]), 1)))
    elif category_type == 'animal':
        for i in range(4, 8):
            train_X.append(events_data[i][:int(t_ratio * events_data[i].shape[0])])
            train_Y.append((i - 4) * np.ones((int(t_ratio * events_data[i].shape[0]), 1)))
            valid_X.append(events_data[i][int(t_ratio * events_data[i].shape[0]):int((t_ratio + v_ratio) * events_data[i].shape[0])])
            valid_Y.append((i - 4) * np.ones((int(v_ratio * events_data[i].shape[0]), 1)))
            test_X.append(events_data[i][int((t_ratio + v_ratio) * events_data[i].shape[0]):])
            test_Y.append((i - 4) * np.ones((int((1 - t_ratio - v_ratio) * events_data[i].shape[0]), 1)))
    elif category_type == 'two':
        indices = [2, 5]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    elif category_type == 'four':
        indices = [2, 3, 5, 6]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    elif category_type == 'two_fruit':
        indices = [2, 3]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    elif category_type == 'two_animal':
        indices = [5, 6]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    elif category_type == 'S1':
        indices = [0, 1, 6, 7]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    elif category_type == 'S2':
        indices = [2, 3, 6, 7]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    elif category_type == 'S3':
        indices = [2, 3, 6, 7]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    elif category_type == 'S4':
        indices = [0, 2, 4, 7]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    elif category_type == 'S5':
        indices = [2, 3, 5, 7]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    elif category_type == 'S6':
        indices = [3, 5, 6, 7]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    elif category_type == 'S7':
        indices = [3, 4, 5, 7]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    elif category_type == 'S8':
        indices = [1, 2, 5, 7]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    elif category_type == 'S9':
        indices = [3, 5, 6, 7]
        for i in range(len(indices)):
            train_X.append(events_data[indices[i]][:int(t_ratio * events_data[indices[i]].shape[0])])
            train_Y.append(i * np.ones((int(t_ratio * events_data[indices[i]].shape[0]), 1)))
            valid_X.append(events_data[indices[i]][
                           int(t_ratio * events_data[indices[i]].shape[0]):int(
                               (t_ratio + v_ratio) * events_data[indices[i]].shape[0])])
            valid_Y.append(i * np.ones((int(v_ratio * events_data[indices[i]].shape[0]), 1)))
            test_X.append(events_data[indices[i]][int((t_ratio + v_ratio) * events_data[indices[i]].shape[0]):])
            test_Y.append(i * np.ones((int((1 - t_ratio - v_ratio) * events_data[indices[i]].shape[0]), 1)))
    else:
        raise TypeError('The classification type is incorrect!')

    train_X = np.concatenate(train_X)
    train_Y = np.concatenate(train_Y)
    test_X = np.concatenate(test_X)
    test_Y = np.concatenate(test_Y)
    
    return train_X, train_Y, test_X, test_Y
