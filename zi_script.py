from data_processing import *
from model_training import train_model_final, get_model_configs_subject
from numpy.random import seed
import tensorflow as tf
from tensorflow.keras.backend import clear_session
import pandas as pd
import numpy as np
from evaluation import convert_to_polar, cut_standing_phase, convert_to_gp
from model_training import custom_rmse

seed(1)
tf.compat.v1.set_random_seed(seed=5)

subjects = np.arange(7, 11)
trials = np.arange(1, 5)

window_size = 80  

# Load Data
path = 'data/evalData/'
headers = ['leftJointPosition', 'rightJointPosition', 'leftJointVelocity', 
        'rightJointVelocity', 'imuGyroX', 'imuGyroY', 'imuGyroZ', 'imuAccX',
        'imuAccY', 'imuAccZ']
method = 'ML'

res = {}
data = import_subject_data(subjects, trials)

for subject in data.keys():
    cur_subject = (int)(subject[-1])
    if cur_subject == 0: cur_subject = 10
    hyperparam_space = {
        'subject': [cur_subject],
        'fold': ['ZIZI'],
        'window_size': [80],
        'model': 'cnn', 
        'cnn': {
        'kernel_size': [20],
        'activation': ['sigmoid']
        },
        'dense': {
            'activation': ['tanh']
        },
        'optimizer': {
            'loss': ['mean_absolute_error'],
            'lr': [0.001],
            'optimizer': ['adam']
        },
        'training': {
            'epochs': [200],
            'batch_size': [128]
        }
    }
    hyperparameter_configs = get_model_configs_subject(hyperparam_space)
    
    subject_rmse = {}
    
    filename = path + f'labeled_AB{cur_subject}_{method}.txt'
    
    eval_data = pd.read_csv(filename)
    eval_data, _ = cut_standing_phase(eval_data)

    #Sliding window for test data
    test_data = eval_data[headers].to_numpy()     
    shape_des = (test_data.shape[0] - window_size +
                1, window_size, test_data.shape[-1])
    strides_des = (
        test_data.strides[0], test_data.strides[0], test_data.strides[1])
    test_data = np.lib.stride_tricks.as_strided(test_data, shape=shape_des,
                                            strides=strides_des)

    #Sliding window for validate data
    labels = eval_data[['leftGaitPhase', 'rightGaitPhase']].to_numpy()
    labels = labels[window_size-1:]
        
    for i in range(4):
        trial_nums = np.arange(i) + 1
        trial_num = 4 - len(trial_nums)
        print(f'\n\nNEW MODEL: SUBJECT {cur_subject} TRIAL_NUM {trial_num}\n')
        for trial in data[subject].keys():
            data[subject][trial] = dict([(key, val) for key, val in data[subject][trial].items() if key not in trial_nums])
    
        model = train_model_final(hyperparam_space['model'], hyperparameter_configs, data)
        
        prediction = model.predict(test_data)
        clear_session()
        true_l_x, true_l_y = convert_to_polar(labels[:, 0])
        true_r_x, true_r_y = convert_to_polar(labels[:, 1])
        ground_truth = np.stack((true_l_x, true_l_y, true_r_x, true_r_y)).T

        l_rmse, r_rmse = custom_rmse(prediction, ground_truth)
        subject_rmse[trial_num] = np.mean((l_rmse, r_rmse))
    res[cur_subject] = subject_rmse
    res_df = pd.DataFrame.from_dict(res)
    print(res_df)
    res_df.to_csv('zi_res.csv')
    
        