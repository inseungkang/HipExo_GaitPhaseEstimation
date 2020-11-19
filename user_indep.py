import pandas as pd
from user_indep_utils import *
import numpy as np
from numpy.random import seed
import tensorflow as tf
import matplotlib.pyplot as plt

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

# Seed Random Number Generators for Reproducibility
seed(1)
tf.compat.v1.set_random_seed(seed=5)

subjects = np.arange(1, 11)

path = 'data/evalData/'
headers = pd.read_csv('data/evalData/headers.txt')
method = 'ML'

v3_sensors = ['leftJointPosition', 'rightJointPosition','leftJointVelocity', 
              'rightJointVelocity', 'imuGyroX', 'imuGyroY', 'imuGyroZ', 
              'imuAccX', 'imuAccY', 'imuAccZ', 'leftWalkMode', 'rightWalkMode', 'leftGaitPhaseX', 
              'leftGaitPhaseY', 'rightGaitPhaseX', 'rightGaitPhaseY']

# Load eval data for all subjects
data_list = {}

for subj in subjects:
    filename = path + f'labeled_AB{subj}_ML.txt'
    data = pd.read_csv(filename, index_col = 0)
    data = data[v3_sensors]
    data_list[f'AB{subj}'] = data

# # List hyperparameters to sweep
# hyperparam_space = {
#     'window_size': [80, 100, 120],
#     'model': 'cnn',
#     'cnn': {
#       'kernel_size': [10, 20, 30],
#       'layers': [2, 3, 4],
#       'activation': ['relu', 'tanh', 'sigmoid'],
#       'output_size': [1, 3, 5]
#     },
#     'dense': {
#         'activation': ['relu', 'tanh', 'sigmoid'],
#         'layers': [(0, 1), (1,10),(1,20),(1,30),(2,10),(2,20),(2,30)]
#     },
#     'optimizer': {
#         'loss': ['mean_absolute_error'],
#         'lr': [0.0001, 0.001, 0.01, 0.1],
#         'optimizer': ['adam', 'sgd', 'rmsprop', 'adagrad']
#     },
#     'training': {
#         'epochs': [200],
#         'batch_size': [128]
#     }
# }

# Final hyperparameter to deploy
hyperparam_space = {
    'window_size': [120],
    'model': 'cnn',
    'cnn': {
      'kernel_size': [20],
      'layers': [2],
      'activation': ['relu'],
      'output_size': [5]
    },
    'dense': {
        'activation': ['tanh'],
        'layers': [(2,30)]
    },
    'optimizer': {
        'loss': ['mean_absolute_error'],
        'lr': [ 0.01],
        'optimizer': ['adagrad']
    },
    'training': {
        'epochs': [200],
        'batch_size': [128]
    }
}


hyperparameter_configs = get_model_configs_independent(hyperparam_space)


# num_samples = 1000
# if num_samples < len(hyperparameter_configs):
#     hyperparameter_configs = np.random.choice(hyperparameter_configs, num_samples)
# print(len(hyperparameter_configs))
# hyperparameter_configs = hyperparameter_configs[:10]


# Train models and evaluate
trial_results, average_results = train_big_models_independent(hyperparam_space['model'], hyperparameter_configs, data_list)
# model, training_history = deploy_big_models_independent(hyperparam_space['model'], hyperparameter_configs, data_list)

# Store results in file
# trial_results.to_csv('trial_results.csv')
average_results.to_csv('average_results_mod_dep.csv')
# model.save('final_model.h5')

# plt.plot(training_history['accuracy'])
# plt.plot(training_history['val_accuracy'])
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()

# plt.plot(training_history['loss'])
# plt.plot(training_history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()