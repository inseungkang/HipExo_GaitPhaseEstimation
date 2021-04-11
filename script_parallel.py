from data_cleaning import *

# segment_data()
# manual_scrap_data(5)
# manual_label_chopped_data(5)

from model_training import *
from data_processing import *
import pandas as pd
from numpy.random import seed
import tensorflow as tf
from joblib import Parallel, delayed


# Seed Random Number Generators for Reproducibility
seed(1)
tf.compat.v1.set_random_seed(seed=5)

subjects = np.arange(1, 11)
trials = np.arange(1, 6)

# NOTE: fold supports 3 ways of folding - ZI (ZI data only), BT (BT data only),
# and ZIBT (ZI+BT for train, BT for validation)
# NOTE: don't put 'lr' in hyperparam_space to indicate default learning rate
# for each optimizers

# CNN Model
def model_parallel(combo):
    
    window_size = combo[0]
    kernel_size = combo[1]
    model_activation = combo[2]
    dense_activation = combo[3]
    optimizer = combo[4]
    learning_rate = combo[5]


    hyperparam_space = {
        'subject': subjects,
        'fold': ['ZIBT'],
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
            'epochs': [200],
            'batch_size': [128]
        }
    }
    hyperparameter_configs = get_model_configs_subject(hyperparam_space)
    data = import_subject_data(subjects, trials)
    trial_results, average_results = train_models_subject(hyperparam_space['model'], hyperparameter_configs, data)
    # train_model_final(hyperparam_space['model'], hyperparameter_configs, data)

    base_path_dir = "/HDD/hipexo/Inseung/Result/"
    trial_save_path = base_path_dir + "trial_results_cnn_ZIBT_window.txt"
    average_save_path = base_path_dir + "average_results_cnn_ZIBT_window.txt"

    trial_results = trial_results.to_csv(sep=' ')
    average_results = average_results.to_csv(sep=' ')

    trial_results = ' '.join([trial_results,"\n"])
    average_results = ' '.join([average_results,"\n"])

    return trial_save_path, trial_results, average_save_path, average_results

run_combos = []
for window_size in [40, 60, 80, 100, 120, 140]:
    for kernel_size in [20]:
        for model_activation in ['tanh']:
            for dense_activation in ['tanh']:
                for optimizer in ['SGD']:
                    for learning_rate in [0.01]:
                            run_combos.append([window_size, kernel_size, model_activation, dense_activation, optimizer, learning_rate])

result = Parallel(n_jobs=-1)(delayed(model_parallel)(combo) for combo in run_combos)
for r in result:
    with open(r[0],"a+") as f:
        f.write(r[1])
    with open(r[2],"a+") as f:
        f.write(r[3])

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

# print(hyperparameter_configs)




# trial_results.to_csv('trial_results.csv')
# average_results.to_csv('average_results.csv')

