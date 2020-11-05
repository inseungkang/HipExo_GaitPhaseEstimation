import pandas as pd
from user_indep_utils import *
import numpy as np
from numpy.random import seed
import tensorflow as tf

# Seed Random Number Generators for Reproducibility
seed(1)
tf.compat.v1.set_random_seed(seed=5)

subjects = np.arange(1, 11)

path = 'data/evalData/'
headers = pd.read_csv('data/evalData/headers.txt')
method = 'ML'

v3_sensors = ['leftJointPosition', 'rightJointPosition','leftJointVelocity', 
              'rightJointVelocity', 'imuGyroX', 'imuGyroY', 'imuGyroZ', 
              'imuAccX', 'imuAccY', 'imuAccZ', 'leftGaitPhaseX', 
			  'leftGaitPhaseY', 'rightGaitPhaseX', 'rightGaitPhaseY']

# Load eval data for all subjects
data_list = {}

for subj in subjects:
	filename = path + f'labeled_AB{subj}_ML.txt'
	data = pd.read_csv(filename, index_col = 0)
	data = data[v3_sensors]
	data_list[f'AB{subj}'] = data

# List hyperparameters to sweep
hyperparam_space = {
	'window_size': [80, 100, 120, 140, 160, 180, 200],
	'model': 'cnn',
	'cnn': {
	  'kernel_size': [10, 20, 30],
	  'activation': ['relu', 'tanh', 'sigmoid']
	},
	'dense': {
		'activation': ['relu', 'tanh', 'sigmoid']
	},
	'optimizer': {
		'loss': ['mean_absolute_error'],
		'lr': [0.0001, 0.001, 0.01, 0.1],
		'optimizer': ['adam', 'sgd', 'rmsprop', 'adagrad']
	},
	'training': {
		'epochs': [150],
		'batch_size': [128]
	}
}
hyperparameter_configs = get_model_configs_independent(hyperparam_space)

# Train models and evaluate
trial_results, average_results = train_big_models_independent(hyperparam_space['model'], hyperparameter_configs, data_list)

# Store results in file
trial_results.to_csv('trial_results.csv')
average_results.to_csv('average_results.csv')