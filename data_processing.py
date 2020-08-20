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
    label1 = pd.read_csv('labels/label_trial1.txt')
    label2 = pd.read_csv('labels/label_trial2.txt')
    label3 = pd.read_csv('labels/label_trial3.txt')
    label4 = pd.read_csv('labels/label_trial4.txt')
    label5 = pd.read_csv('labels/label_trial5.txt')
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


def cut_features_bulk_cnn(data_list, cutting_indices_list):
    # take a list of data and a list of cutting_indices
    # cut the data off at the standing section
    # Store the files into the ../features folder
    ct = 1
    for data, cutting_indices in zip(data_list, cutting_indices_list):
        for i in range(math.floor((len(cutting_indices)/2))):
            features = data.iloc[cutting_indices[i*2]:cutting_indices[(i*2)+1]+1]
            filename = f'features/cnn_feature{ct}.txt'
            features.to_csv(filename, index=False)
            ct = ct+1


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


