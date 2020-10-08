from data_cleaning import *

# segment_data()
# manual_scrap_data(5)
# manual_label_chopped_data(5)

from model_training import *
from data_processing import *
import pandas as pd
from numpy.random import seed
import tensorflow as tf

# Seed Random Number Generators for Reproducibility
seed(1)
tf.compat.v1.set_random_seed(seed=5)

subjects = np.arange(1, 3)
trials = np.arange(1, 5)

# NOTE: fold supports 3 ways of folding - ZI (ZI data only), BT (BT data only),
# and ZIBT (ZI+BT for train, BT for validation)
# NOTE: don't put 'lr' in hyperparam_space to indicate default learning rate
# for each optimizers

# CNN Model
# NOTE: don't put 'subject' in hyperparam_space for independent model
hyperparam_space = {
    'fold': ['BT'],
    'window_size': [100],
    'model': 'cnn',
    'cnn': {
      'kernel_size': [10],
      'activation': ['relu']
    },
    'dense': {
        'activation': ['tanh']
    },
    'optimizer': {
        'loss': ['mean_absolute_error'],
        'lr': [0.0001],
        'optimizer': ['adam']
    },
    'training': {
        'epochs': [1],
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

hyperparameter_configs = get_model_configs_independent(hyperparam_space)
# print(hyperparameter_configs)

data = import_subject_data(subjects, trials)

trial_results, average_results = train_models_independent(hyperparam_space['model'], hyperparameter_configs, data)


# trial_results, average_results = train_models_subject(hyperparam_space['model'], hyperparameter_configs, data)

trial_results.to_csv('trial_results.csv')
average_results.to_csv('average_results.csv')

# train_model_final(hyperparam_space['model'], hyperparameter_configs, data)