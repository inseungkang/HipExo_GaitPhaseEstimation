import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    data_list, left_joint_positions, right_joint_positions = [], [], []
    for data in data_all:
        # drop the 32nd column which only contains NaN values
        data.dropna(axis=1, inplace=True) 
        # rename the columns
        data.columns = columns_list
        # only keep the 10 sensors data columns
        data = data[sensors]
        data_list.append(data)
        # create joing position lists
        left_joint_positions.append(data['leftJointPosition'])
        right_joint_positions.append(data['rightJointPosition'])

    return data_list, left_joint_positions, right_joint_positions


def find_local_maximas(data):
    # Peak detection using scipy.signal.find_peaks()
    
    data = data.rolling(10).mean() # smooth out the data
    peaks, _ = find_peaks(data) # find all extremas in the data

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

# Produce 1 label file for each trial and store them in ../labels folder
data, left_joint_positions, right_joint_positions = import_data()
for i in range(5):
    filename = "labels/label_trial{}.txt".format(i+1)
    left_x, left_y = label_vectors(left_joint_positions[i])
    right_x, right_y = label_vectors(right_joint_positions[i])
    label_df = pd.DataFrame({'leftGaitPhaseX': left_x, 'leftGaitPhaseY': left_y, 'rightGaitPhaseX': right_x, 'rightGaitPhaseY': right_y})
    label_df.to_csv(filename, index=False)

plot_graph_uni(left_joint_positions[0])
plot_graph_bi(left_joint_positions[0], right_joint_positions[0])
# plt.show()
