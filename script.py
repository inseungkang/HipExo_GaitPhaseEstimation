from utils import *
import pickle
import pandas as pd

path = 'data/strokeData/'
headers = pd.read_csv('data/strokeData/field_v3.txt')
subject = 'ST05'
assistance_levels = ['L0R0', 'L0R1', 'L1R1', 'L2R1', 'L2R2', 'L3R1']
window_size = 80

f = open('data/strokeData/ST05/windowed_L0R0.pkl', 'rb')
dataset = pickle.load(f)
split_windowed_dataset(dataset, 'leftGaitPhase', 'splittest')
# split_dataset(ex_data, 'leftJointPosition', 'test')
# print(ex_data.head())
# print(ex_data.columns)
# manual_scrap_data(ex_data, 'test')
exit()
# hyperparam_space = {
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
#         'lr': [0.0001],
#         'optimizer': ['adam']
#     },
#     'training': {
#         'epochs': [1],
#         'batch_size': [128]
#     }
# }

# hyperparameter_configs = get_model_configs_independent(hyperparam_space)

# data = import_subject_data(subjects, trials)

trial_results, average_results = train_models_independent(hyperparam_space['model'], hyperparameter_configs, data)


# trial_results, average_results = train_models_subject(hyperparam_space['model'], hyperparameter_configs, data)

trial_results.to_csv('trial_results.csv')
average_results.to_csv('average_results.csv')

# train_model_final(hyperparam_space['model'], hyperparameter_configs, data)

# ccw = np.loadtxt('data/evalData/AB10_CCW_TBE.txt', skiprows=1)
# cw = np.loadtxt('data/evalData/AB10_CW_TBE.txt', skiprows=1)
# tbe = np.concatenate([ccw, cw], axis = 0)
# np.savetxt('data/evalData/AB10_TBE.txt', tbe)