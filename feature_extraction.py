import math
import numpy as np
import pandas as pd
from labeling import find_local_maximas, left_joint_positions, right_joint_positions


columns_list = pd.read_csv('data/columns.txt', header=None).transpose().values.tolist()[0]
columns = ['leftJointPosition', 'rightJointPosition', 'leftJointVelocity', 'rightJointVelocity', 'imuGyroX', 'imuGyroY', 'imuGyroZ', 'imuAccX', 'imuAccY', 'imuAccZ']
extractions = ['Min', 'Max', 'Std', 'Avg', 'Last']


# Import data
data1 = pd.read_csv('data/trial_1.txt', sep=" ", header=None)
data2 = pd.read_csv('data/trial_2.txt', sep=" ", header=None)
data3 = pd.read_csv('data/trial_3.txt', sep=" ", header=None)
data4 = pd.read_csv('data/trial_4.txt', sep=" ", header=None)
data5 = pd.read_csv('data/trial_5.txt', sep=" ", header=None)

data_all = [data1, data2, data3, data4, data5]
data_list = []

for data in data_all:
    # drop the 32nd column with only NaN values
    data.dropna(axis=1, inplace=True) 
    # rename the columns
    data.columns = columns_list
    # only keep the 10 needed columns
    data_cleaned = data[columns]
    data_list.append(data_cleaned)


# Import labels
label1 = pd.read_csv('labels/label_trial1.txt')
label2 = pd.read_csv('labels/label_trial2.txt')
label3 = pd.read_csv('labels/label_trial3.txt')
label4 = pd.read_csv('labels/label_trial4.txt')
label5 = pd.read_csv('labels/label_trial5.txt')
label_all = [label1, label2, label3, label4, label5]


window_sizes = [40, 60, 80, 100, 120, 140]

# create a list of feature names
feature_columns = []
for extraction in extractions:
    for column in columns:
        feature_columns.append(column+extraction)


def extract_features(data, window_size, features, labels):
    # Extracts the features from data based on the list of window sizes
    # Combine the labels and the features
    for i in range(window_size, data.shape[0]+1):
        data_window = data[i-window_size:i]
        feature = data_window.min()
        feature = feature.append(data_window.max(), ignore_index=True)
        feature = feature.append(data_window.std(), ignore_index=True)
        feature = feature.append(data_window.mean(), ignore_index=True)
        feature = feature.append(data_window.iloc[window_size-1], ignore_index=True)
        features_length = len(features)
        features.loc[features_length] = feature.tolist()
    features[labels.columns] = labels.iloc[window_size-1:].values
    return features

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

    cuts = maximas[diff>(2*np.std(diff)+np.mean(diff))]
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
    return peaks_ix


def cut_features(features, cutting_indices):
    # cut off the rows in between the cutting_indices
    features_cut = pd.DataFrame(columns=features.columns)
    for i in range(math.floor((len(cutting_indices)/2))):
        features_cut = features_cut.append(features.iloc[cutting_indices[i*2]:cutting_indices[(i*2)+1]+1])
    return features_cut


# Create 1 feature file for each trial and each window size combination
# Store the files at ../features folder
for i in range(5):
    for window_size in window_sizes:
        features = pd.DataFrame(columns=feature_columns)
        features_df = extract_features(data_list[i], window_size, features, label_all[i])
        filename = 'features/trial{}_winsize{}.txt'.format(i+1, window_size)
        features_df.to_csv(filename, index=False)
        print(f'Feature extraction completed for trial {i+1} window size {window_size}.')

# Create a cleaned data file for each trial and window sizes
# Store the file at ../features_clean folder
for trial_num in range(1, 6):
    cutting_indices = find_cutting_indices(left_joint_positions[trial_num-1], right_joint_positions[trial_num-1])
    cutting_indices_arr = np.array(cutting_indices)
    for i in window_sizes:
        # import features
        read_filename = 'features/trial{}_winsize{}.txt'.format(trial_num, i)
        features = pd.read_csv(read_filename)
        
        cut_ix = cutting_indices_arr - (i-1) # adjust the cut_ix based on the window size
        
        features_cut = cut_features(features, cut_ix)
        save_filename = 'features_clean/trial{}_winsize{}_clean.txt'.format(trial_num, i)
        features_cut.to_csv(save_filename, index=False)

