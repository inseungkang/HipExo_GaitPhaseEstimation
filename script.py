from data_cleaning import *

# segment_data()
# manual_scrap_data(5)
# manual_label_chopped_data(5)

from model_training import *
from data_processing import *
import pandas as pd
from numpy.random import seed
import tensorflow as tf
from data_cleaning import manual_scrap_data

# Seed Random Number Generators for Reproducibility
seed(1)
tf.compat.v1.set_random_seed(seed=5)

subjects = np.arange(1, 2)
trials = np.arange(1, 5)

# data_list = []
# for i in [1, 2]:
#     # For Mannually Labeling Data
#     filename = path + f'AB{subject}_{method}_{i}.txt'
#     data = pd.read_csv(filename, skiprows=1, sep=" ")
#     data = data.dropna(axis=1)
#     data.columns = headers.columns.str.replace(' ', '')
#     data = data.loc[:,~data.columns.duplicated()]
#     data_list.append(data)
#     print(data.shape)
# data = pd.concat(data_list)
# print(data.shape)
# data.to_csv(path + f'AB{subject}_{method}.txt')

# NOTE: fold supports 3 ways of folding - ZIBT (ZI data only), BTBT (BT data only),
# and ZIBT (ZI+BT for train, BT for validation), ZIZI (Train and validate on ZI)
# NOTE: don't put 'lr' in hyperparam_space to indicate default learning rate
# for each optimizers

# CNN Model
# NOTE: don't put 'subject' in hyperparam_space for independent model
hyperparam_space = {
    'subject': subjects,
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

# # MLP Model
# hyperparam_space = {
#     'subject': subjects,
#     'fold': ['ZIBT'],
#     'window_size': [20],
#     'model': 'mlp',
#     'dense': {
#         'num_layers': [1],
#         'num_nodes': [5],
#         'activation': ['relu']
#     },
#     'optimizer': {
#         'loss': ['mean_absolute_error'],
#         'lr': [0.001],
#         'optimizer': ['adam', 'sgd', 'rmsprop', 'adagrad']
#     },
#     'training': {
#         'epochs': [2],
#         'batch_size': [128]
#     }
# }

# # LSTM Model
# hyperparam_space = {
#     'subject': subjects,
#     'fold': ['BT'],
#     'window_size': [40],
#     'model': 'lstm',
#     'lstm': {
#       'units': [20],
#       'activation': ['tanh']
#     },
#     'dense': {
#         'activation': ['tanh']
#     },
#     'optimizer': {
#         'loss': ['mean_absolute_error'],
#         'optimizer': ['adam']
#     },
#     'training': {
#         'epochs': [2],
#         'batch_size': [128]
#     }
# }

hyperparameter_configs = get_model_configs_subject(hyperparam_space)
# print(hyperparameter_configs)

data = import_subject_data(subjects, trials)

# trial_results, average_results = train_models_independent(hyperparam_space['model'], hyperparameter_configs, data)

# trial_results.to_csv('trial_results.csv')
# average_results.to_csv('average_results.csv')
