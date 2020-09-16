from data_processing import *
import re
import math
from keras.optimizers import SGD, Adam, RMSprop
from neural_network import train_nn

sensors = ['leftJointPosition', 'rightJointPosition', 'leftJointVelocity',
           'rightJointVelocity', 'imuGyroX', 'imuGyroY', 'imuGyroZ', 'imuAccX',
           'imuAccY', 'imuAccZ']
window_sizes = [40]

def find_standing_phase(data):
    ''' 
    Input: 1D array of joint position data
    Output: 1D array of indices representing the standing phase segments in the
    format of [start, end, start, ... end]
    '''

    diff = np.abs(np.diff(data))
    threshold = 0.008
    diff[diff <= threshold], diff[diff > threshold] = 0, 1

    # use string pattern matching to find the start and end indices of the
    # standing phases
    diff_str = ''.join(str(x) for x in diff.astype(int))
    begin = [m.start() for m in re.finditer(r'10{200}', diff_str)]
    end = [m.end() for m in re.finditer(r'0{200}1', diff_str)]

    if (np.min(end) < np.min(begin)): begin.append(0)
    if (np.max(end) < np.max(begin)): end.append(len(data))

    standing_indices = np.sort(np.hstack([begin, end]))
    return standing_indices

data = import_data(sensors)
left_joint_positions, right_joint_positions = extract_joint_positions(data)

# Labeling data
y_list = []
for trial_num in range(5):
    left_stand_ix = find_standing_phase(left_joint_positions[trial_num])
    right_stand_ix = find_standing_phase(right_joint_positions[trial_num])
    stand_ix = np.mean([left_stand_ix, right_stand_ix], axis=0).astype(int)
    y = np.ones(len(data[trial_num]))
    for i in range(math.floor(len(stand_ix)/2)):
        y[stand_ix[2*i]:stand_ix[2*i+1]] = 0
    y_list.append(pd.DataFrame(y, columns=['y']))

    plt.plot(left_joint_positions[trial_num])
    plt.plot(right_joint_positions[trial_num])
    plt.vlines(stand_ix, -1, 1)
    plt.show()

    
#feature_extraction(data, y_list, window_sizes, sensors, cut=False)

# Best Parameters
best_window_size = [40]
best_num_layer = [3]
best_num_node = [15]
best_optimizer = [Adam()]

# train NN for stand/walk classification
error = train_nn(best_window_size, best_num_layer, best_num_node, best_optimizer, mode='classification')


# Plot prediction vs real label for trial 1
y_true = y_list[0]
y_pred = pd.read_csv(
    "predictions/classification_wsize40_3layers_15nodes_optimizer1_trial1.txt")
plt.plot(y_true, label="true")
plt.plot(y_pred, label="predicted")
plt.legend()
plt.title("Stand/Walk Classification")
plt.yticks([0,1], labels=['stand', 'walk'])
plt.show()
