from data_processing import *
from convolutional_nn import *
from lstm_modules import *
from tensorflow.keras.optimizers import Adam

sensors = ['leftJointPosition', 'rightJointPosition', 'leftJointVelocity',
           'rightJointVelocity', 'imuGyroX', 'imuGyroY', 'imuGyroZ', 'imuAccX',
           'imuAccY', 'imuAccZ']

# Produce 1 label file for each trial and store them in ../labels folder
data = import_data(sensors)

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

# Creat a list of cut_indicies for each trial
cut_indicies_list = []
for i in range(5):
    cut_indicies_list.append(find_cutting_indices(left_joint_positions[i], 
    right_joint_positions[i]))

# Cut the standing data and store files into ../features folder
data_list = cnn_cut_data(data, cut_indicies_list)

###################### Training Neural Network ##################
window_sizes = np.arange(20, 221, 20)

train_cnn(data_list, window_sizes, [Adam()])
plot_err(window_sizes * 5)
