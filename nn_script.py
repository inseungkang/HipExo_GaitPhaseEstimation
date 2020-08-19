from data_processing import *
from neural_network import train_nn, plot_time_series, plot_err
from keras.optimizers import SGD, Adam, RMSprop

sensors = ['leftJointPosition', 'rightJointPosition', 'leftJointVelocity',
           'rightJointVelocity', 'imuGyroX', 'imuGyroY', 'imuGyroZ', 'imuAccX',
           'imuAccY', 'imuAccZ']
window_sizes = [40]

# Produce 1 label file for each trial and store them in ../labels folder
data = import_data(sensors)
left_joint_positions, right_joint_positions = extract_joint_positions(data)
for i in range(5):
    filename = "labels/label_trial{}.txt".format(i+1)
    left_x, left_y = label_vectors(left_joint_positions[i])
    right_x, right_y = label_vectors(right_joint_positions[i])
    label_df = pd.DataFrame({'leftGaitPhaseX': left_x, 'leftGaitPhaseY': left_y,
                             'rightGaitPhaseX': right_x, 'rightGaitPhaseY': right_y})
    label_df.to_csv(filename, index=False)


###################### Training Neural Network ##################

# Feature extraction and cut the standing data sections
labels = import_labels()
extract_and_cut(data, labels, window_sizes, sensors)


# Swept Parameters
window_sizes = np.array([40, 60, 80, 100, 120, 140])
num_layers = np.array([1, 2, 3, 4])
num_nodes = np.arange(5, 51, 5)
learning_rate = 0.005
opt_SGD = SGD(lr=learning_rate)
opt_ADAM = Adam(lr=learning_rate)
opt_RMSprop = RMSprop(lr=learning_rate)
optimizers = [opt_SGD, opt_ADAM, opt_RMSprop]

# Best Parameters
best_window_size = [40]
best_num_layer = [3]
best_num_node = [15]
best_optimizer = [opt_ADAM]

# train a bilateral model using the best params
error = train_nn(best_window_size, best_num_layer,
                 best_num_node, best_optimizer, mode='bi')
plot_time_series(best_winsow_size)

