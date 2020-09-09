import math
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.signal import find_peaks, peak_prominences, peak_widths
import warnings
warnings.filterwarnings("ignore")

def import_data(sensors):
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


def import_labels():
    # Import labels
    label1 = pd.read_csv('labels/label_trial1_raw.txt')
    label2 = pd.read_csv('labels/label_trial2_raw.txt')
    label3 = pd.read_csv('labels/label_trial3_raw.txt')
    label4 = pd.read_csv('labels/label_trial4_raw.txt')
    label5 = pd.read_csv('labels/label_trial5_raw.txt')
    label_all = [label1, label2, label3, label4, label5]
    return label_all


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


def plot_graph_bi(data_left, data_right):
    # Plot hip positions and gait cycles for both sides in one graph
    
    fig, (gait_cycle, hip_position) = plt.subplots(2,1, sharex=True, figsize=(16, 8))
    
    # plot hip positions
    hip_position.set_xlim(0, data_left.shape[0])
    hip_position.plot(data_left, 'blue', label='Left')
    hip_position.plot(data_right, 'red', alpha=0.7, label='right')
    peaks_left = find_local_maximas(data_left)
    peaks_right = find_local_maximas(data_right)
    hip_position.plot(peaks_left, data_left[peaks_left], 'bo')
    hip_position.plot(peaks_right, data_right[peaks_right], 'ro')
    hip_position.set_title('Hip Positions')
    hip_position.legend()

    # plot gait cycles
    gait_cycle.set_ylim(0, 1)
    xl, xr, yl, yr = [0], [0], [0], [0]
    for peak in peaks_left:
        xl.append(peak)
        xl.append(peak+1)
        yl.append(1)
        yl.append(0)
    xl.append(data_left.shape[0])
    yl.append(0)
    gait_cycle.plot(xl, yl, 'b-')
    for peak in peaks_right:
        xr.append(peak)
        xr.append(peak+1)
        yr.append(1)
        yr.append(0)
    xr.append(data_right.shape[0])
    yr.append(0)
    gait_cycle.plot(xr, yr, 'r-', alpha=0.7)
    gait_cycle.set_title('Gait Cycles')


def plot_graph_uni(data):
    # Plot hip positions and gait cycles for one side
    # data is a 1-D hip position array
    fig, (gait_cycle, hip_position) = plt.subplots(2,1, sharex=True, figsize=(10, 5))
    
    # plot hip positions
    hip_position.set_xlim(0, data.shape[0])
    hip_position.plot(data, 'blue')
    peaks = find_local_maximas(data)
    hip_position.plot(peaks, data[peaks], 'bx')
    hip_position.set_title('Hip Positions')

    # plot gait cycles
    gait_cycle.set_ylim(0, 1)
    x, y = [0], [0]
    for peak in peaks:
        x.append(peak)
        x.append(peak+1)
        y.append(1)
        y.append(0)
    x.append(data.shape[0])
    y.append(0)
    gait_cycle.plot(x, y, 'y-')
    gait_cycle.set_title('Gait Cycles')


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

def label_vectors_raw(data):
    # Create label vectors based on joint positions
    # Does not convert to polar coordinates
    maximas = find_local_maximas(data)
    y = pd.Series(np.nan, index=range(0, data.shape[0]))
    for maxima in maximas:
        y[maxima] = 1
        y[maxima+1] = 0
    y.interpolate(inplace=True)
    y.fillna(0, inplace=True)
    return y


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


def cut_features(features, cutting_indices):
    # cut off the rows in between the cutting_indices
    features_cut = pd.DataFrame(columns=features.columns)
    for i in range(math.floor((len(cutting_indices)/2))):
        features_cut = features_cut.append(
            features.iloc[cutting_indices[i*2]:cutting_indices[(i*2)+1]+1])
    return features_cut


def feature_extraction(data_list, labels_list, window_sizes, sensors, cut=True):
    # Extracts the features from data based on the list of window sizes
    # Combine the labels and the features
    # Cut the standing portion of the data out

    extractions = ['Min', 'Max', 'Std', 'Avg', 'Last']
    
    # create a list of feature names
    feature_columns = []
    for extraction in extractions:
        for sensor in sensors:
            feature_columns.append(sensor+extraction)
    
    if cut: left_joint, right_joint = extract_joint_positions(data_list)

    for ix, data, labels in zip(range(1,6), data_list, labels_list):
        # find the list indices to cut the data
        if cut: cutting_indices = find_cutting_indices(left_joint[ix-1], 
            right_joint[ix-1])
        for window_size in window_sizes:
            features = pd.DataFrame(columns=feature_columns)
            if cut: cut_ix = cutting_indices - (window_size-1)
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
            if cut: featuress = cut_features(features, cut_ix)
            # Save features as files
            filename = f'features/trial{ix}_winsize{window_size}.txt'
            if cut:
                filename = f'features/cut_trial{ix}_winsize{window_size}.txt'
            features.to_csv(filename, index=False)

############## functions specificly for CNN ###################

def cnn_cut_data(data_list, cutting_indices_list):
    '''
    Params:
        data_list: list of DataFrames
        cutting_indicces_list: list of integers representing the start and end
        of each new data object in the format of [start, end, ... start, end]
    Return:
        list of DataFrames containing data objects
    Split each data in data_list into multiple DatFrames based on the
    cutting_indices_list.
    '''
    features_list = []
    for data, cutting_indices in zip(data_list, cutting_indices_list):
        for i in range(math.floor((len(cutting_indices)/2))):
            features = data.iloc[cutting_indices[i*2]:cutting_indices[(i*2)+1]+1]
            features_list.append(features)
    return features_list

def cnn_extract_images(data_list, window_sizes):
    '''
    Params:
        data_list: list of DataFrames
        window_sizes: list of integers
    Writes a features file and a labels files to ../features for each 
    combination of data and window size
    '''
    for ix, data in enumerate(data_list):
        # split features(X) and labels(y)
        data = data.to_numpy()
        label_columns = np.arange(10, 14)
        X = np.delete(data, label_columns, axis=1)
        y = data[:, label_columns]

        for window_size in window_sizes:
            filename = f'features/cnn_trial{ix+1}_winsize{window_size}'
            # Store labels(y) as .npy files
            # y.shape = (m, 4)
            label_arr = y[window_size-1:, :]
            np.save(filename+'_y',label_arr)

            # Extract images from X and store as .npy files
            # X.shape = (m, win_size, 10, 1)
            image_arr = None
            for i in range(window_size, X.shape[0]+1):
                image = X[i-window_size:i]
                if image_arr is None: image_arr = image
                else: image_arr = np.dstack((image_arr, image))
            image_arr = np.transpose(image_arr, (2, 0, 1))[..., np.newaxis]
            np.save(filename+'_X', image_arr)

def norm_matrix(data_list):
    # Output: norm_matrix (2, 10) - features-wise min and max across data in data_list
    min_full = np.zeros((1, 10))
    max_full = np.zeros((1, 10))
    for data in data_list:
        data = data.to_numpy()
        trial_X = data[:, :-2]
        trial_min = np.reshape(trial_X.min(
            axis=0, keepdims=True), (1, 10))
        trial_max = np.reshape(trial_X.max(
            axis=0, keepdims=True), (1, 10))
        min_full = np.vstack((min_full, trial_min))
        max_full = np.vstack((max_full, trial_max))
    min_full = min_full[1:, :].min(axis=0)
    max_full = max_full[1:, :].max(axis=0)
    norm_matrix = np.vstack((min_full, max_full))
    return norm_matrix
    
    
def cnn_extract_features(data_list, window_size, testing_trial):
    norm = norm_matrix(data_list)
    full_min = np.reshape(norm[0, :], (1, 1, 10))
    full_max = np.reshape(norm[1, :], (1, 1, 10))
    
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
            trial_X = data[:, :-2]
            trial_Y = data[:, -2:]

            #Sliding window
            shape_des = (trial_X.shape[0] - window_size + 1, window_size, trial_X.shape[-1])
            strides_des = (
                trial_X.strides[0], trial_X.strides[0], trial_X.strides[1])
            trial_X = np.lib.stride_tricks.as_strided(trial_X, shape=shape_des,
                                                    strides=strides_des)
            trial_Y = trial_Y[window_size-1:]
            trial_Y_x = np.cos(trial_Y * math.pi * 2)
            trial_Y_y = np.sin(trial_Y * math.pi * 2)
            trial_Y = np.hstack((trial_Y_x.reshape(
                trial_Y_x.shape[0], 2), trial_Y_y.reshape(trial_Y_y.shape[0], 2)))
            
            trial_X = (trial_X - full_min)/(full_max-full_min)

            X_test = np.concatenate([X_test, trial_X], axis=0)
            Y_test = np.concatenate([Y_test, trial_Y], axis=0)

            X_test = X_test[1:, :, :]
            Y_test = Y_test[1:, :]

            data_out['X_test'] = X_test
            data_out['y_test'] = Y_test
            
        else:
            # Generate Training Data
            trial_X = data[:, :-2]
            trial_Y = data[:,-2:] #raw gp%

            #Sliding window
            shape_des = (trial_X.shape[0] - window_size + 1, window_size, trial_X.shape[-1])
            strides_des = (trial_X.strides[0], trial_X.strides[0], trial_X.strides[1])
            trial_X = np.lib.stride_tricks.as_strided(trial_X, shape=shape_des,
                        strides=strides_des)
            trial_Y = trial_Y[window_size-1:]
            trial_Y_x = np.cos(trial_Y * (math.pi * 2))
            trial_Y_y = np.sin(trial_Y * (math.pi * 2))
            trial_Y = np.hstack((trial_Y_x, trial_Y_y))
            # trial_X.shape = (N, winsize, 10), trial_Y.shape = (N, 4)

            trial_X = (trial_X - full_min)/(full_max-full_min)

            X_train = np.concatenate([X_train, trial_X], axis=0)
            Y_train = np.concatenate([Y_train, trial_Y], axis=0)

    X_train = X_train[1:, :, :]
    Y_train = Y_train[1:, :]
    data_out['X_train'] = X_train
    data_out['y_train'] = Y_train

    return data_out