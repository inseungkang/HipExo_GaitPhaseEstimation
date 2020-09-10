from keras.metrics import RootMeanSquaredError
from data_processing import *
from convolutional_nn import *
import pandas as pd
import numpy as np
import math
import os
import gc
gc.enable()
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error
import statistics
import random
# from sets import Set
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM, Conv1D, Flatten, Activation
from keras.optimizers import Adam
from keras import backend as K 
from keras.utils import multi_gpu_model



sensors = ['leftJointPosition', 'rightJointPosition', 'leftJointVelocity',
           'rightJointVelocity', 'imuGyroX', 'imuGyroY', 'imuGyroZ', 'imuAccX',
           'imuAccY', 'imuAccZ']

# Produce 1 label file for each trial and store them in ../labels folder
data = import_data(sensors)

left_joint_positions, right_joint_positions = extract_joint_positions(data)

# for i in range(5):
#     filename = "labels/label_trial{}_raw.txt".format(i+1)
#     # left_x, left_y = label_vectors(left_joint_positions[i])
#     # right_x, right_y = label_vectors(right_joint_positions[i])
#     # label_df = pd.DataFrame({'leftGaitPhaseX': left_x, 'leftGaitPhaseY': left_y,
#     #                          'rightGaitPhaseX': right_x, 'rightGaitPhaseY': right_y})
#     # label_df.to_csv(filename, index=False)
#     left_y = label_vectors_raw(left_joint_positions[i])
#     right_y = label_vectors_raw(right_joint_positions[i])
#     labels = pd.DataFrame({'leftGaitPhase': left_y, 'rightGaitPhase': right_y})
#     labels.to_csv(filename, index=False)

# Combine the data and the labels
labels = import_labels()
for d, l in zip(data, labels):
    d[l.columns] = l


# Creat a list of cut_indicies for each trial
cut_indicies_list = []
for i in range(5):
    cut_indicies_list.append(find_cutting_indices(left_joint_positions[i],
                                                  right_joint_positions[i]))

# Cut the standing data
data_list = cnn_cut_data(data, cut_indicies_list)
norm = norm_matrix(data_list)
full_min = np.reshape(norm[0, :], (1, 1, 10))
full_max = np.reshape(norm[1, :], (1, 1, 10))
errs = []
for window_size in [160, 180]:
    conv_kernel = 10
    error_list = []
    for testing_trial in range(1, 11):
        X_test = np.zeros((1, window_size, 10))
        Y_test = np.zeros((1, 4))
        X_train = np.zeros((1, window_size, 10))
        Y_train = np.zeros((1, 4))
        for i, data in enumerate(data_list):
            data = data.to_numpy()
            if i+1 == testing_trial:
                # Generate Testing Data
                # raw gp%, not (x,y)
                trial_X = data[:, :-2].astype(np.float)
                trial_Y = data[:, -2:].astype(np.float)

                #Sliding window
                shape_des = (trial_X.shape[0] - window_size +
                            1, window_size, trial_X.shape[-1])
                strides_des = (
                    trial_X.strides[0], trial_X.strides[0], trial_X.strides[1])
                trial_X = np.lib.stride_tricks.as_strided(trial_X, shape=shape_des,
                                                        strides=strides_des)
                trial_Y = trial_Y[window_size-1:]
                trial_Y_x = np.cos(trial_Y * (math.pi * 2))
                trial_Y_y = np.sin(trial_Y * (math.pi * 2))
                trial_Y = np.hstack((trial_Y_x.reshape(
                    trial_Y_x.shape[0], 2), trial_Y_y.reshape(trial_Y_y.shape[0], 2)))

                trial_X = (trial_X - full_min)/(full_max-full_min)

                X_test = np.concatenate([X_test, trial_X], axis=0)
                Y_test = np.concatenate([Y_test, trial_Y], axis=0)

                X_test = X_test[1:, :, :]
                Y_test = Y_test[1:, :]
                print("Testing on: ", X_test.shape, Y_test.shape)
        
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
        print("Training on: ", X_train.shape, Y_train.shape)

        # save_model_string = "../Model Checkpoints/" + testing_trial + "_" + str(window_size) + "_cnn_small_filter.hdf5"
        # save_model_string = "../Production Models New/" + testing_trial + "_gait_phase_independent_model_" + str(window_size) + ".h5"
        # model_checkpoint_callback = ModelCheckpoint(save_model_string, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)

        model_dep = Sequential()
        model_dep.add(Conv1D(X_train.shape[-1], conv_kernel, input_shape=(X_train.shape[1], X_train.shape[-1]), trainable=False))
        model_dep.add(Conv1D(X_train.shape[-1], window_size - conv_kernel + 1, trainable=False))
        model_dep.add(Activation('relu'))
        model_dep.add(Flatten())
        model_dep.add(Dense(4, activation='tanh'))
        model_dep.compile(loss='mean_squared_error',
                        optimizer='adam', metrics=RootMeanSquaredError())

        # model_dep = load_model(save_model_string)

        # model_weights_path = "../Model Checkpoints/" + "cnn_locked_conv_complex_weights_" + str(window_size) + ".hdf5"
        # model_weights_path = "../Production Models/" + testing_trial + "_gait_phase_independent_model.hdf5"
        # model_weights_path = "../Production Models New/" + testing_trial + "_gait_phase_independent_model_" + str(window_size) + ".hdf5"
        # model_dep.load_weights(model_weights_path)
        # model = multi_gpu_model(model, gpus=2)
        # model_dep.compile(loss='mean_squared_error', optimizer='adam')
        # model.summary()
        # exit()
        # model_dep.fit(X_train, Y_train, epochs=100, batch_size=128, verbose=0, validation_split=0.2, shuffle=True, callbacks= [model_checkpoint_callback, early_stopping_callback])
        # model_dep.fit(X_train, Y_train, epochs=100, batch_size=128, verbose=0, validation_split=0.2, shuffle=True, callbacks= [early_stopping_callback])
        history = model_dep.fit(X_train, Y_train, epochs=100, batch_size=128, verbose=0, validation_data=(X_test, Y_test), shuffle=True, callbacks= [early_stopping_callback])
        plot_learning_curve(history, testing_trial, window_size)
        plt.show()
        y_hat = model_dep.predict(X_test)
        left_rmse, right_rmse = custom_rmse(Y_test, y_hat)
        error_list.append(np.mean((left_rmse, right_rmse)))
        gc.collect()
    
    print("\n\nAverage across all trials: ", window_size)
    # mean_error = statistics.mean(error_list)
    mean_error = np.nanmean(error_list)
    errs = np.append(errs, mean_error)
np.savetxt('err.txt', errs)

