import pandas as pd
import matplotlib.pyplot as plt
from data_processing import *
from model_training import *

def convert_to_polar(gp):
    theta = gp * 2 * np.pi
    x = np.cos(theta)
    y = np.sin(theta)
    return x, y

def calculate_truth_and_pred(data):
    maximas = find_local_maximas(data['leftJointPosition'])
    
    l_x_true, l_y_true = label_vectors(data['leftJointPosition'])
    l_x_pred, l_y_pred = convert_to_polar(data['leftGaitPhase'])
    r_x_true = r_y_true = r_x_pred = r_y_pred = pd.Series(0, index=data.index)
    true = pd.concat([l_x_true, l_y_true, r_x_true, r_y_true], axis=1)
    true.columns = ['l_x_true', 'l_y_true', 'r_x_true', 'r_y_true']

    pred = pd.concat([l_x_pred, l_y_pred, r_x_pred, r_y_pred], axis=1)
    pred.columns = ['l_x_pred', 'l_y_pred', 'r_x_pred', 'r_y_pred']
    
    data = pd.concat([data, true, pred], axis=1)
    data = data.iloc[maximas[0]:maximas[-1]+1, :].reset_index()
    
    return data

def find_mode_transition_cycles(data):
    transition_cycles = []
    mode_diff = [0] + np.diff(data['leftWalkMode'])
    mode_transition = [i for i, v in enumerate(mode_diff) if v != 0]
    maximas = find_local_maximas(data['leftJointPosition'])
    for point in mode_transition:
        closest = (min(maximas, key=lambda x : abs(x-point)))
        transition_cycles.append((closest , maximas[maximas.index(closest) + 1]))
    # plt.plot(mode_transition, [1]*len(mode_transition), 'r*')
    # plt.vlines(transition_cycles, -1.5, 0.5, 'y')
    # plt.plot(data['leftJointPosition'])
    # plt.plot(data['leftWalkMode'])
    # plt.show()
    return transition_cycles

############################### Evaluation Script ############################

# Load Data
path = 'data/evalData/'
headers = pd.read_csv(path + 'headers.txt')
data = pd.read_csv(path + 'log_cw.txt', skiprows=1, sep=" ")

# Clean data format
data = data.dropna(axis=1)
data.columns = headers.columns.str.replace(' ', '')
data = data.loc[:,~data.columns.duplicated()]

# add 4 ground truth columns and 4 prediction columns at the end of data
data = calculate_truth_and_pred(data)

# Calculate the overall rmse
overall_rmse = custom_rmse(data.iloc[:, -8:-4].to_numpy(), 
                           data.iloc[:, -4:].to_numpy())

# Find indices for mode transition gait cycles and calculate transition rmse
transition_cycles = find_mode_transition_cycles(data)
transition_data = pd.DataFrame([])
for indeces in transition_cycles:
    transition_data = transition_data.append(data.iloc[indeces[0]:indeces[1]+1, :])
transition_rmse = custom_rmse(transition_data.iloc[:, -8:-4].to_numpy(), 
                              transition_data.iloc[:, -4:].to_numpy())

# Calculate mode specific rmse
ntransition_data = data.drop(index=transition_data.index)
modes = ntransition_data['leftWalkMode'].unique()
mode_rmse = {}
for mode in modes:
    mode_data = ntransition_data[ntransition_data['leftWalkMode'] == mode]
    mode_rmse[f'mode {mode}'] = custom_rmse(mode_data.iloc[:, -8:-4].to_numpy(), 
                                  mode_data.iloc[:, -4:].to_numpy())

print(overall_rmse, transition_rmse)
print(mode_rmse)
