import pandas as pd
from user_indep_utils import *
import numpy as np
from numpy.random import seed
import tensorflow as tf
from joblib import Parallel, delayed

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

def model_parallel(combo):
	window_size = combo[0]
	kernel_size = combo[1]
	model_activation = combo[2]
	dense_activation = combo[3]
	optimizer = combo[4]
	learning_rate = combo[5]

	data_list = {}

	for subj in subjects:
		filename = path + f'labeled_AB{subj}_ML.txt'
		data = pd.read_csv(filename, index_col = 0)
		data = data[v3_sensors]
		data_list[f'AB{subj}'] = data

	# List hyperparameters to sweep
	hyperparam_space = {
		'window_size': [window_size],
		'model': 'cnn',
		'cnn': {
		  'kernel_size': [kernel_size],
		  'activation': [model_activation]
		},
		'dense': {
			'activation': [dense_activation]
		},
		'optimizer': {
			'loss': ['mean_absolute_error'],
			'lr': [learning_rate],
			'optimizer': [optimizer]
		},
		'training': {
			'epochs': [150],
			'batch_size': [128]
		}
	}
	hyperparameter_configs = get_model_configs_independent(hyperparam_space)
	trial_results, average_results = train_models_independent(hyperparam_space['model'], hyperparameter_configs, data_list)

	base_path_dir = "/HDD/hipexo/Inseung/Result/"
	trial_save_path = base_path_dir + "trial_results_IND_learning_rate.txt"
	average_save_path = base_path_dir + "average_results_IND_learning_rate.txt"

	trial_results = trial_results.to_csv(sep=' ')
	average_results = average_results.to_csv(sep=' ')

	trial_results = ' '.join([trial_results,"\n"])
	average_results = ' '.join([average_results,"\n"])

	return trial_save_path, trial_results, average_save_path, average_results

run_combos = []
for window_size in [100]:
    for kernel_size in [10]:
        for model_activation in ['tanh']:
            for dense_activation in ['tanh']:
                for optimizer in ['adagrad']:
                    for learning_rate in [0.0001, 0.001, 0.01]:
                            run_combos.append([window_size, kernel_size, model_activation, dense_activation, optimizer, learning_rate])

result = Parallel(n_jobs=-1)(delayed(model_parallel)(combo) for combo in run_combos)
for r in result:
    with open(r[0],"a+") as f:
        f.write(r[1])
    with open(r[2],"a+") as f:
        f.write(r[3])