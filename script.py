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
tf.random.set_seed(seed=5)

subjects = np.arange(1, 2)
trials = np.arange(1, 6)

# # CNN Model
# hyperparam_space = {
#     'subject': subjects,
#     'fold': ['BT'],
#     'window_size': [100],
#     'model': 'cnn',
#     'cnn': {
#       'kernel_size': [10],
#       'activation': ['relu']
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

# # MLP Model
# hyperparam_space = {
#     'subject': subjects,
#     'fold': ['BT'],
#     'window_size': [20],
#     'model': 'mlp',
#     'dense': {
#         'num_layers': [1],
#         'num_nodes': [5],
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

# # LSTM Model
# hyperparam_space = {
#     'subject': subjects,
#     'fold': ['ZI', 'BT'],
#     'window_size': [40],
#     'model': 'lstm',
#     'lstm': {
#       'units': [20],
#       'activation': ['tanh']
#     },
#     'dense': {
#         'activation': ['relu']
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

data = import_subject_data(subjects, trials)

trial_results, average_results = train_models_subject(hyperparam_space['model'], hyperparameter_configs, data)
trial_results.to_csv('trial_results.csv')
average_results.to_csv('average_results.csv')