import math
import glob
import re, sys
import pandas as pd
import numpy as np
import scipy
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, peak_prominences, peak_widths
import warnings
warnings.filterwarnings("ignore")

columns = ['lJPos', 'rJPos', 'lJVel', 'rJVel', 'lJTorque', 'rJTorque',
           'eulerX', 'eulerY', 'eulerZ', 'gyroX', 'gyroY', 'gyroZ', 'accX', 'accY', 'accZ',
           'batt', 'cpu', 'mem', 'lBttn', 'rBttn', 'time', 'lJVelFilt', 'rJVelFilt',
           'lJPosReset', 'rJPosReset', 'lGC', 'rGC', 'stand', 'lCmdTorque', 'rCmdTorque',
           'lRecvTorque', 'rRecvTorque', 'lStanceSwing', 'rStanceSwing', 'nWalk', 'lWalk', 'rWalk', 'none']
sensors = ['lJPos', 'rJPos', 'lJVel',
           'rJVel', 'gyroX', 'gyroY', 'gyroZ', 'accX',
           'accY', 'accZ', 'nWalk']
gait_phase = ['leftGaitPhaseX', 'leftGaitPhaseY', 'rightGaitPhaseX', 'rightGaitPhaseY']

def segment_data():
    """load and segment out data from each circuit and cutting out standing data
    """
    for subject in range(10, 11):
        for file_path in glob.glob(f'data/raw/AB{subject:02d}*.txt'):
            data = pd.read_csv(file_path, sep=" ", header=None)
            data.columns = columns
            lJPos, rJPos = extract_joint_positions([data])
            stand = find_standing_phase(lJPos[0])
            stand = np.append(stand, len(lJPos[0]) - 1)
            diff = np.abs(np.diff(stand))
            diff_ix = [i for i, v in enumerate(diff) if v > 3000]
            cut_ix = []
            for i in diff_ix:
                cut_ix.append(stand[i])
                cut_ix.append(stand[i+1])
            data_cut = []
            for i in range(math.floor((len(cut_ix)/2))):
                segment = data.iloc[cut_ix[i*2]
                    :cut_ix[(i*2)+1]+1]
                data_cut.append(segment)
                
            for i, segment in enumerate(data_cut[-5:]):
                save_file_path = file_path[:9] + '/' + file_path[10:-4] + f'_{i+1}'
                # np.save(save_file_path, segment)
            # print(len(data_cut))
            # print(file_path)
            # print(len(data_cut))
    #         print(stand)
    #         print(diff)
    #         print(cut_ix)
    #         print("")
            plt.figure(figsize=(10, 5))
            plt.plot(lJPos[0])
            plt.plot(rJPos[0], 'r')
            plt.vlines(cut_ix[-10:], -1, 1)
            plt.title(file_path)
    plt.show()


def find_standing_phase(data):
    ''' 
    Input: 1D array of joint position data
    Output: 1D array of indices representing the standing phase segments in the
    format of [start, end, start, ... end]
    '''

    diff = np.abs(np.diff(data))
    threshold = 0.0058
    diff[diff <= threshold], diff[diff > threshold] = 0, 1

    # use string pattern matching to find the start and end indices of the
    # standing phases
    diff_str = ''.join(str(x) for x in diff.astype(int))
    begin = [m.start() for m in re.finditer(r'10{97}', diff_str)]
    end = [m.end() for m in re.finditer(r'0{97}1', diff_str)]

    if (np.min(end) < np.min(begin)):
        begin.append(0)
    if (np.max(end) < np.max(begin)):
        end.append(len(data))

    standing_indices = np.sort(np.hstack([begin, end]))
    return standing_indices

def import_subject_data(subject_list, trial_list):
    """ import data and organize them based on subject and direction (CW/CCW)

    Args:
        subject_list (list[int]): a list of subject numbers

    Returns:
        dict{dict{dict{list[DataFrames]}}}: ex: data = {'AB01': {'ZICW': {1: [dataframe,
        ...], 2: [dataframe, ...]}, 'BTCCW': {1: [dataframe, ...]}, 'AB02': {...} }
    """
    
    data = {}
    for subject in subject_list:
        subject_data = {}
        for condition in ['ZI', 'BT']:
            for direction in ['CW', 'CCW']:
                trial_dict = {}
                for trial in trial_list:
                    file_path = f'data/AB{subject:02d}/labeled*_' + direction + '_' + condition + '_' + str(trial) + '*'
                    # Read data
                    data_list = []
                    for file in sorted(glob.glob(file_path)):
                        trial_data = np.loadtxt(file)
                        trial_data = pd.DataFrame(trial_data, columns=sensors+gait_phase)
                        data_list.append(trial_data)
                    trial_dict[trial] = data_list
                subject_data[condition+direction] = trial_dict
        data[f'AB{subject:02d}'] = subject_data
    return data


def import_data():
    """imports all 5 trials of data, take only the columns corresponds to 
    sensors in the sensors list. Format them into dataframe and put them in a
    list.
    
    Returns:
        list[Dataframes]: 5 trials of data of shape (M, 10)
    """

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
    """Label the data, and combine the data with the label columns
    Args:
        data (list[Dataframes]): 5 trials of data of shape (M, 10)
    Returns:
        list[Dataframes]: 5 trials of data of shape (M, 14)
    """
    
    left_joint_positions, right_joint_positions = extract_joint_positions(data)

    labels = []
    for i in range(5):
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
    """Cut the standing phase of data off and split data into segments

    Args:
        data (list[Dataframes]): 5 trials of data of shape (M, 14)

    Returns:
        list[Dataframes]: 10 segments of data of shape (N, 14)
    """
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
    """Extracts the left and right joint position data from each trial and put
    them in a list

    Args:
        data_all (list[Dataframes]): data containing sensors information

    Returns:
        left_joint_positions (list[Series]): left joint positions from each trials
        right_joint_positions (list[Series]): right joint positions from each trials
    """
    left_joint_positions, right_joint_positions = [], []
    for data in data_all:
        # create joing position lists
        left_joint_positions.append(data['lJPos'])
        right_joint_positions.append(data['rJPos'])
    return left_joint_positions, right_joint_positions


def find_local_maximas(joint_positions):
    """find the maximas in joint positions

    Args:
        joint_positions (Series): joint position time seire data

    Returns:
        [ndarray]: a list of local maximas for joint position
    """
    # Peak detection using scipy.signal.find_peaks()

    # joint_positions = joint_positions.rolling(10).mean()  # smooth out the joint positions
    peaks, _ = find_peaks(joint_positions)  # find all extremas in the joint positions

    # find a list of prominences for all extremas
    prominences = peak_prominences(joint_positions, peaks)[0]
    width = peak_widths(joint_positions, peaks)[0]

    # find maximas
    # Constrains:   prominance of peaks > median + variance of prominances
    #               height of peaks > mean(joint_positions)
    #               distance between peaks > 100 samples
    #               width of peak < mean + 4*std of width
    maximas, _ = find_peaks(joint_positions, 
                            prominence=np.median(prominences)+np.var(prominences), 
                            distance=150)
    return maximas.tolist()


def label_vectors(joint_positions):
    """generates the gait phase from a joint position time series data and
    converts to polar coordinates

    Args:
        joint_positions (Series): joint position time seire data

    Returns:
        gait_phase_x (float): the polar coordinate x gait phase
        gait_phase_y (float): the polar coordinate y gait phase
    """
    # Create label vectors based on joint positions and convert to polar coordinates
    maximas = find_local_maximas(joint_positions)
    y = pd.Series(np.nan, index=range(0, joint_positions.shape[0]))
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
    """generates a list of indices where the data should be cut off

    Args:
        left_data (Series): left joint position
        right_data (Series): right joint position

    Returns:
        ndarray: indices where the data should be cut off at
    """
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
    """feature extraction for regular MLP

    Args:
        data_list (list[DataFrames]): list of all data of shape (M, 14)
        window_size (int): window size
        testing_trial (int): between 1 - 10, represents the test trial

    Returns:
        dictionary: X_train, X_test - (M, 50); y_train, y_test - (M, 4)
    """
    X_train = np.zeros((1, 50))
    Y_train = np.zeros((1, 4))
    data_out = {}
    for i, data in enumerate(data_list):
        # mode = data['nWalk']
        trial_X = data.iloc[:, :-4]
        # trial_X = trial_X.drop(['nWalk'])
        trial_Y = data.iloc[:, -4:]
        if i+1 == testing_trial:
            feature_extracted_data = pd.DataFrame()
            for ix, column in enumerate(trial_X.columns):
                single_column = trial_X.iloc[:, i].values
                shape_des = single_column.shape[:-1] + \
                    (single_column.shape[-1] - window_size + 1, window_size)
                strides_des = single_column.strides + \
                    (single_column.strides[-1],)

                sliding_window = np.lib.stride_tricks.as_strided(
                    single_column, shape=shape_des, strides=strides_des)
                sliding_window_df = pd.DataFrame(sliding_window)

                min_series = sliding_window_df.min(axis=1)
                max_series = sliding_window_df.max(axis=1)
                mean_series = sliding_window_df.mean(axis=1)
                std_series = sliding_window_df.std(axis=1)
                last_series = sliding_window_df.iloc[:, -1]

                feature_extracted_data = pd.concat([feature_extracted_data, round(min_series, 4), round(max_series, 4), round(
                    mean_series, 4), round(std_series, 4), round(last_series, 4)], axis=1, ignore_index=True)
            Y_test = trial_Y.iloc[window_size-1:].to_numpy()
            data_out['X_test'] = feature_extracted_data
            data_out['y_test'] = Y_test
        else:
            feature_extracted_data = pd.DataFrame()
            for ix, column in enumerate(trial_X.columns):
                single_column = trial_X.iloc[:, i].values
                shape_des = single_column.shape[:-1] + \
                    (single_column.shape[-1] - window_size + 1, window_size)
                strides_des = single_column.strides + (single_column.strides[-1],)

                sliding_window = np.lib.stride_tricks.as_strided(
                    single_column, shape=shape_des, strides=strides_des)
                sliding_window_df = pd.DataFrame(sliding_window)

                min_series = sliding_window_df.min(axis=1)
                max_series = sliding_window_df.max(axis=1)
                mean_series = sliding_window_df.mean(axis=1)
                std_series = sliding_window_df.std(axis=1)
                last_series = sliding_window_df.iloc[:, -1]

                feature_extracted_data = pd.concat([feature_extracted_data, round(min_series, 4), round(max_series, 4), round(
                    mean_series, 4), round(std_series, 4), round(last_series, 4)], axis=1, ignore_index=True)
            trial_Y = trial_Y.iloc[window_size-1:].to_numpy()
            X_train = np.concatenate([X_train, feature_extracted_data], axis=0)
            Y_train = np.concatenate([Y_train, trial_Y], axis=0)
    data_out['X_train'] = X_train
    data_out['y_train'] = Y_train
    return data_out


def cnn_extract_features(data_list, window_size, testing_trial):
    """feature extraction for CNN and LSTM
    Args:
        data_list (list[DataFrames]): list of all data of shape (M, 14)
        window_size (int): window size
        testing_trial (int): between 1 - 10, represents the test trial
    Returns:
        dictionary: X_train, X_test - (M, window_size, 10); y_train, y_test - (M, 4)
    """
    
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


def nn_extract_features_subject(subject_data, window_size, test_trial, fold):
    
    testing_data = []
    training_data = []  
    if fold == 'ZI':
        for condition in ['ZICW','ZICCW']:
            for trial in subject_data[condition].keys():
                training_data.extend(subject_data[condition][trial])
        for condition in ['BTCW', 'BTCCW']:
            testing_data.extend(subject_data[condition][test_trial])
    elif fold == 'BT':
        for condition in ['BTCW','BTCCW']:
            for trial in subject_data[condition].keys():
                if trial == test_trial:
                    testing_data.extend(subject_data[condition][trial])
                else:
                    training_data.extend(subject_data[condition][trial])
    elif fold == 'ZIBT':
        for condition in subject_data.keys():
            for trial in subject_data[condition].keys():
                if condition in ['BTCW', 'BTCCW']:
                    if trial == test_trial:
                        testing_data.extend(subject_data[condition][trial])
                    else:
                        training_data.extend(subject_data[condition][trial])
                else:
                    training_data.extend(subject_data[condition][trial])
   
    X_test = np.zeros((1, 50))
    Y_test = np.zeros((1, 4))
    X_train = np.zeros((1, 50))
    Y_train = np.zeros((1, 4))
    mode = np.zeros((1))
    data_out = {}
    
    # Generate testing data
    for i, data in enumerate(testing_data):
        if 'nWalk' in data.columns:
            nWalk = data['nWalk']
            data = data.drop(columns='nWalk')
 
        trial_X = data.iloc[:, :-4]
        trial_Y = data.iloc[:, -4:]

        feature_extracted_data = pd.DataFrame()
        for ix, column in enumerate(trial_X.columns):
            single_column = trial_X.iloc[:, ix].values
            shape_des = single_column.shape[:-1] + \
                (single_column.shape[-1] - window_size + 1, window_size)
            strides_des = single_column.strides + (single_column.strides[-1],)

            sliding_window = np.lib.stride_tricks.as_strided(
                single_column, shape=shape_des, strides=strides_des)
            sliding_window_df = pd.DataFrame(sliding_window)

            min_series = sliding_window_df.min(axis=1)
            max_series = sliding_window_df.max(axis=1)
            mean_series = sliding_window_df.mean(axis=1)
            std_series = sliding_window_df.std(axis=1)
            last_series = sliding_window_df.iloc[:, -1]

            feature_extracted_data = pd.concat([feature_extracted_data, round(min_series, 4), round(max_series, 4), round(
                mean_series, 4), round(std_series, 4), round(last_series, 4)], axis=1, ignore_index=True)
        if nWalk is not None:
            nWalk = nWalk.iloc[window_size-1:].to_numpy()
            mode = np.concatenate([mode, nWalk])
        trial_Y = trial_Y.iloc[window_size-1:].to_numpy()
        X_test = np.concatenate([X_test, feature_extracted_data], axis=0)
        Y_test = np.concatenate([Y_test, trial_Y], axis=0)
    X_test = X_test[1:, :]
    Y_test = Y_test[1:, :]
    data_out['X_test'] = X_test
    data_out['y_test'] = Y_test
    if len(mode) > 1:
        mode = mode[1:]
        data_out['mode'] = mode
    
    # Generate training data
    for i, data in enumerate(training_data):
        if 'nWalk' in data.columns:
            data = data.drop(columns='nWalk')
        trial_X = data.iloc[:, :-4]
        trial_Y = data.iloc[:, -4:]

        feature_extracted_data = pd.DataFrame()
        for ix, column in enumerate(trial_X.columns):
            single_column = trial_X.iloc[:, ix].values
            shape_des = single_column.shape[:-1] + \
                (single_column.shape[-1] - window_size + 1, window_size)
            strides_des = single_column.strides + (single_column.strides[-1],)

            sliding_window = np.lib.stride_tricks.as_strided(
                single_column, shape=shape_des, strides=strides_des)
            sliding_window_df = pd.DataFrame(sliding_window)

            min_series = sliding_window_df.min(axis=1)
            max_series = sliding_window_df.max(axis=1)
            mean_series = sliding_window_df.mean(axis=1)
            std_series = sliding_window_df.std(axis=1)
            last_series = sliding_window_df.iloc[:, -1]

            feature_extracted_data = pd.concat([feature_extracted_data, round(min_series, 4), round(max_series, 4), round(
                mean_series, 4), round(std_series, 4), round(last_series, 4)], axis=1, ignore_index=True)
        trial_Y = trial_Y.iloc[window_size-1:].to_numpy()
        X_train = np.concatenate([X_train, feature_extracted_data], axis=0)
        Y_train = np.concatenate([Y_train, trial_Y], axis=0)
    X_train = X_train[1:, :]
    Y_train = Y_train[1:, :]
    data_out['X_train'] = X_train
    data_out['y_train'] = Y_train
    return data_out


def cnn_extract_features_subject(subject_data, window_size, test_trial, fold):
    testing_data = []
    training_data = []  
    if fold == 'ZI':
        for condition in ['ZICW','ZICCW']:
            for trial in subject_data[condition].keys():
                training_data.extend(subject_data[condition][trial])
        for condition in ['BTCW', 'BTCCW']:
            testing_data.extend(subject_data[condition][test_trial])
    elif fold == 'BT':
        for condition in ['BTCW','BTCCW']:
            for trial in subject_data[condition].keys():
                if trial == test_trial:
                    testing_data.extend(subject_data[condition][trial])
                else:
                    training_data.extend(subject_data[condition][trial])
    elif fold == 'ZIBT':
        for condition in subject_data.keys():
            for trial in subject_data[condition].keys():
                if condition in ['BTCW', 'BTCCW']:
                    if trial == test_trial:
                        testing_data.extend(subject_data[condition][trial])
                    else:
                        training_data.extend(subject_data[condition][trial])
                else:
                    training_data.extend(subject_data[condition][trial])
   
    X_test = np.zeros((1, window_size, 10))
    Y_test = np.zeros((1, 4))
    X_train = np.zeros((1, window_size, 10))
    Y_train = np.zeros((1, 4))
    mode = np.zeros((1))
    data_out = {}
    
    # Generate testing data
    for i, data in enumerate(testing_data):
        if 'nWalk' in data.columns:
            nWalk = data['nWalk']
            data = data.drop(columns='nWalk')

        data = data.to_numpy()
        
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
        if nWalk is not None:
            nWalk = nWalk.iloc[window_size-1:].to_numpy()
            mode = np.concatenate([mode, nWalk])
    
    X_test = X_test[1:, :, :]
    Y_test = Y_test[1:, :]

    data_out['X_test'] = X_test
    data_out['y_test'] = Y_test

    if len(mode) > 1:
        mode = mode[1:]
        data_out['mode'] = mode
    
    # Generate Training Data
    for i, data in enumerate(training_data):
        if 'nWalk' in data.columns:
            data = data.drop(columns='nWalk')
        
        data = data.to_numpy()
        
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

        X_train = np.concatenate([X_train, trial_X], axis=0)
        Y_train = np.concatenate([Y_train, trial_Y], axis=0)

    X_train = X_train[1:, :, :]
    Y_train = Y_train[1:, :]
    data_out['X_train'] = X_train
    data_out['y_train'] = Y_train
    return data_out