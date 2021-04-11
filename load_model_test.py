import tensorflow as tf
import pandas as pd
import numpy as np
from evaluation import convert_to_polar, cut_standing_phase, convert_to_gp
from model_training import custom_rmse

window_size = 80  

# Load Data
path = 'data/evalData/'
headers = ['leftJointPosition', 'rightJointPosition', 'leftJointVelocity', 
        'rightJointVelocity', 'imuGyroX', 'imuGyroY', 'imuGyroZ', 'imuAccX',
        'imuAccY', 'imuAccZ']
subject = 1
method = 'ML'
filename = path + f'labeled_AB{subject}_{method}.txt'
data = pd.read_csv(filename)
data, _ = cut_standing_phase(data)

#Sliding window for test data
test_data = data[headers].to_numpy()     
shape_des = (test_data.shape[0] - window_size +
            1, window_size, test_data.shape[-1])
strides_des = (
    test_data.strides[0], test_data.strides[0], test_data.strides[1])
test_data = np.lib.stride_tricks.as_strided(test_data, shape=shape_des,
                                        strides=strides_des)

#Sliding window for validate data
validate_data = data[['leftGaitPhase', 'rightGaitPhase']].to_numpy()
validate_data = validate_data[window_size-1:]

for i in range(1, 5):
    model = tf.keras.models.load_model(f'model{i}.h5')
    prediction = model.predict(test_data)

    true_l_x, true_l_y = convert_to_polar(validate_data[:, 0])
    true_r_x, true_r_y = convert_to_polar(validate_data[:, 1])
    ground_truth = np.stack((true_l_x, true_l_y, true_r_x, true_r_y)).T

    l_rmse, r_rmse = custom_rmse(prediction, ground_truth)

    print(np.mean((l_rmse, r_rmse)))
  
model = tf.keras.models.load_model(f'model{i}.h5')
prediction = model.predict(test_data)

true_l_x, true_l_y = convert_to_polar(validate_data[:, 0])
true_r_x, true_r_y = convert_to_polar(validate_data[:, 1])
ground_truth = np.stack((true_l_x, true_l_y, true_r_x, true_r_y)).T

l_rmse, r_rmse = custom_rmse(prediction, ground_truth)

print(np.mean((l_rmse, r_rmse)))

# import matplotlib.pyplot as plt
# pred_gp_l = convert_to_gp(prediction[:, 0], prediction[:, 1])
# plt.plot(validate_data[:, 0])
# plt.plot(pred_gp_l, alpha=0.7)
# plt.show()

# trials = np.arange(1, 11)
# window_size = 20
# loss_per_trial = []
# for test_trial_num in trials:
#     data = cnn_train_test_split(test_trial_num, window_size)
#     loss_per_trial.append(model.evaluate(
#             data['X_test'], data['y_test']))
# loss_mean = np.mean(loss_per_trial) * 100
# print('Window Size: {} \Loss: {:.2f}%'.format(
#     window_size, loss_mean))