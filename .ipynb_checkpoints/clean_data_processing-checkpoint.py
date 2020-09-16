import math
import pandas as pd
import numpy as np
import scipy
from scipy.signal import find_peaks, peak_prominences, peak_widths
import warnings
warnings.filterwarnings("ignore")

sensors = ['leftJointPosition', 'rightJointPosition', 'leftJointVelocity',
           'rightJointVelocity', 'imuGyroX', 'imuGyroY', 'imuGyroZ', 'imuAccX',
           'imuAccY', 'imuAccZ']


def import_data():
    # Read data
    columns = pd.read_csv('data/columns.txt', header=None)
    data1 = pd.read_csv('data/trial_1.txt', sep=" ", header=None)
    data2 = pd.read_csv('data/trial_2.txt', sep=" ", header=None)
    data3 = pd.read_csv('data/trial_3.txt', sep=" ", header=None)
    data4 = pd.read_csv('data/trial_4.txt', sep=" ", header=None)
    data5 = pd.read_csv('data/trial_5.txt', sep=" ", header=None)

    # Format data
    data_all = [data1, data2, data3, data4, data5]
    columns_list = columns.transpose().values.tolist()[0]
    data_list = []
    for data in data_all:
        # drop the 32nd column which only contains NaN values
        data.dropna(axis=1, inplace=True)
        # rename the columns
        data.columns = columns_list
        # only keep the 10 sensors data columns
        data = data[sensors]
        data_list.append(data)

    return data_list


def label_data(data):
    left_joint_positions, right_joint_positions = extract_joint_positions(data)

    labels = []
    for i in range(5):
        filename = "labels/label_trial{}.txt".format(i+1)
        left_x, left_y = label_vectors(left_joint_positions[i])
        right_x, right_y = label_vectors(right_joint_positions[i])
        label_df = pd.DataFrame({'leftGaitPhaseX': left_x, 'leftGaitPhaseY': left_y,
                                 'rightGaitPhaseX': right_x, 'rightGaitPhaseY': right_y})
        labels.append(label_df)

    # Combine the data and the labels
    for d, l in zip(data, labels):
        d[l.columns] = l

    return data


def cut_data(data):
    left_joint_positions, right_joint_positions = extract_joint_positions(data)

    # Creat a list of cut_indicies for each trial
    cut_indicies_list = []
    for i in range(5):
        cut_indicies_list.append(find_cutting_indices(left_joint_positions[i],
                                                      right_joint_positions[i]))

    features_list = []
    for data, cutting_indices in zip(data, cut_indicies_list):
        for i in range(math.floor((len(cutting_indices)/2))):
            features = data.iloc[cutting_indices[i*2]
                :cutting_indices[(i*2)+1]+1]
            features_list.append(features)
    return features_list


def extract_joint_positions(data_all):
    left_joint_positions, right_joint_positions = [], []
    for data in data_all:
        # create joing position lists
        left_joint_positions.append(data['leftJointPosition'])
        right_joint_positions.append(data['rightJointPosition'])
    return left_joint_positions, right_joint_positions


def find_local_maximas(data):
    # Peak detection using scipy.signal.find_peaks()

    data = data.rolling(10).mean()  # smooth out the data
    peaks, _ = find_peaks(data)  # find all extremas in the data

    # find a list of prominences for all extremas
    prominences = peak_prominences(data, peaks)[0]
    width = peak_widths(data, peaks)[0]

    # find maximas
    # Constrains:   prominance of peaks > median + variance of prominances
    #               height of peaks > mean(data)
    #               distance between peaks > 100 samples
    #               width of peak < mean + 4*std of width
    maximas, _ = find_peaks(data, prominence=np.median(prominences)+np.var(prominences), height=np.mean(data), distance=100,
                            wlen=np.mean(width)+4*np.std(width))

    return maximas


def label_vectors(data):
    # Create label vectors based on joint positions and convert to polar coordinates
    maximas = find_local_maximas(data)
    y = pd.Series(np.nan, index=range(0, data.shape[0]))
    for maxima in maximas:
        y[maxima] = 1
        y[maxima+1] = 0
    y.interpolate(inplace=True)
    y.fillna(0, inplace=True)
    y_theta = y * 2 * np.pi
    gait_phase_x = np.cos(y_theta)
    gait_phase_y = np.sin(y_theta)
    return gait_phase_x, gait_phase_y


def find_cutting_indices(left_data, right_data):
    # takes the left and right joint position arrays as input
    # returns a list of indices representing the starting and ending indices of training data to keep [start, end, start, end, ...]
    left_maximas = find_local_maximas(left_data)
    right_maximas = find_local_maximas(right_data)

    maximas = np.concatenate((left_maximas, right_maximas))
    maximas = np.sort(maximas)

    diff = []
    for i in range(maximas.shape[0]-1):
        diff.append(maximas[i+1]-maximas[i])
    diff.append(0)

    cuts = maximas[diff > (2*np.std(diff)+np.mean(diff))]
    # Starting from peak 2
    peaks_ix = [maximas[1]]

    for cut in cuts:
        stand_ix = np.where(maximas == cut)[0][0]
        peaks_ix.append(maximas[stand_ix-1])
        if stand_ix+2 < maximas.shape[0]:
            peaks_ix.append(maximas[stand_ix+2])
        else:
            return peaks_ix

    peaks_ix.append(maximas[maximas.shape[0]-2])
    return np.array(peaks_ix)


def nn_extract_features(data_list, window_size, testing_trial):
    # TODO: fix this method for sliding windows of NN

    # Extracts the features from data based on the list of window sizes
    # Combine the labels and the features
    # Cut the standing portion of the data out
    extractions = ['Min', 'Max', 'Std', 'Avg', 'Last']

    # create a list of feature names
    feature_columns = []
    for extraction in extractions:
        for sensor in sensors:
            feature_columns.append(sensor+extraction)

    left_joint, right_joint = extract_joint_positions(data_list)

    for ix, data in enumerate(data_list):
        # find the list indices to cut the data
        if cut:
            cutting_indices = find_cutting_indices(left_joint[ix-1],
                                                   right_joint[ix-1])
        features = pd.DataFrame(columns=feature_columns)
        if cut:
            cut_ix = cutting_indices - (window_size-1)
        for i in range(window_size, data.shape[0]+1):
            data_window = data[i-window_size:i]
            feature = data_window.min()
            feature = feature.append(data_window.max(), ignore_index=True)
            feature = feature.append(data_window.std(), ignore_index=True)
            feature = feature.append(data_window.mean(), ignore_index=True)
            feature = feature.append(data_window.iloc[window_size-1],
                                     ignore_index=True)
            features_length = len(features)
            features.loc[features_length] = feature.tolist()

        # Combine the features with the labels
        features[labels.columns] = labels.iloc[window_size-1:].values
        # Cut features as the cut_ix
        if cut:
            featuress = cut_data(features)
    return data_out


def cnn_extract_features(data_list, window_size, testing_trial):

    X_test = np.zeros((1, window_size, 10))
    Y_test = np.zeros((1, 4))
    X_train = np.zeros((1, window_size, 10))
    Y_train = np.zeros((1, 4))
    data_out = {}
    for i, data in enumerate(data_list):
        data = data.to_numpy()
        if i+1 == testing_trial:
            # Generate Testing Data
            # raw gp%, not (x,y)
            trial_X = data[:, :-4]
            trial_Y = data[:, -4:]

            #Sliding window
            shape_des = (trial_X.shape[0] - window_size +
                         1, window_size, trial_X.shape[-1])
            strides_des = (
                trial_X.strides[0], trial_X.strides[0], trial_X.strides[1])
            trial_X = np.lib.stride_tricks.as_strided(trial_X, shape=shape_des,
                                                      strides=strides_des)
            trial_Y = trial_Y[window_size-1:]

            X_test = np.concatenate([X_test, trial_X], axis=0)
            Y_test = np.concatenate([Y_test, trial_Y], axis=0)

            X_test = X_test[1:, :, :]
            Y_test = Y_test[1:, :]

            data_out['X_test'] = X_test
            data_out['y_test'] = Y_test

        else:
            # Generate Training Data
            trial_X = data[:, :-4]
            trial_Y = data[:, -4:]  # raw gp%

            #Sliding window
            shape_des = (trial_X.shape[0] - window_size +
                         1, window_size, trial_X.shape[-1])
            strides_des = (
                trial_X.strides[0], trial_X.strides[0], trial_X.strides[1])
            trial_X = np.lib.stride_tricks.as_strided(trial_X, shape=shape_des,
                                                      strides=strides_des)
            trial_Y = trial_Y[window_size-1:]

            X_train = np.concatenate([X_train, trial_X], axis=0)
            Y_train = np.concatenate([Y_train, trial_Y], axis=0)

    X_train = X_train[1:, :, :]
    Y_train = Y_train[1:, :]
    data_out['X_train'] = X_train
    data_out['y_train'] = Y_train

    return data_out
