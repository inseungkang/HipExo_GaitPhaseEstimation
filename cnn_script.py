from data_processing import *
from convolutional_nn import *

sensors = ['leftJointPosition', 'rightJointPosition', 'leftJointVelocity',
           'rightJointVelocity', 'imuGyroX', 'imuGyroY', 'imuGyroZ', 'imuAccX',
           'imuAccY', 'imuAccZ']

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

# Combine the data and the labels
labels = import_labels()
for d, l in zip(data, labels):
    d[l.columns] = l

# Creat a list of cut_indicies for each trial
cut_indicies_list = []
for i in range(5):
    cut_indicies_list.append(find_cutting_indices(left_joint_positions[i], 
    right_joint_positions[i]))

# Cut the standing data and store files into ../features folder
cut_features_bulk_cnn(data, cut_indicies_list)

###################### Training Neural Network ##################



