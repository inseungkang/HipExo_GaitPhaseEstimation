from utils import *
import pickle
import pandas as pd
import numpy as np
from tensorflow.initializers import he_uniform
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, Activation, Flatten, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from model_training import custom_rmse

path = 'data/strokeData/'
headers = pd.read_csv('data/strokeData/field_v3.txt')
subject = 'ST05'
assistance_levels = ['L0R0', 'L0R1', 'L1R1', 'L2R1', 'L2R2', 'L3R1']

# Get training dataset
with open(path+subject+'/training.pkl', 'rb') as f:
	training_set = pickle.load(f)
	f.close()
 
#Get testing dataset
with open(path+subject+'/testing.pkl', 'rb') as f:
	testing_set = pickle.load(f)
	f.close()

# Shuffle Data
rng = np.random.default_rng(seed=5)
N = training_set['X'].shape[0]
N_test = testing_set['X'].shape[0]
shuffle_indices = np.arange(N)
shuffle_indices_test = np.arange(N_test)
rng.shuffle(shuffle_indices)
rng.shuffle(shuffle_indices_test)
X_train = training_set['X'][shuffle_indices]
Y_train = training_set['Y'][shuffle_indices]
X_test = testing_set['X'][shuffle_indices_test]
Y_test = testing_set['Y'][shuffle_indices_test]

# Index out proper columns for ground truth
gt_cols = training_set['ground_truth_cols']
target_cols = ['leftGaitPhaseX', 'leftGaitPhaseY', 'rightGaitPhaseX', 'rightGaitPhaseY']
target_col_ndx = [gt_cols.index(col) for col in target_cols]
Y_train = Y_train[:, target_col_ndx]
test_cols = testing_set['ground_truth_cols']
test_col_ndx = [test_cols.index(col) for col in target_cols]
Y_test = Y_test[:, test_col_ndx]

window_size = training_set['window_size']
n_features = len(training_set['feature_cols'])

# Create CNN model
kernel_size = 20
n_epochs = 100
batch_size = 128
optimizer = SGD(learning_rate=0.01)
loss = 'mean_absolute_error'


# model = Sequential()
# model.add(BatchNormalization(input_shape=(window_size, n_features)))
# model.add(Conv1D(10,
# 			kernel_size,
# 			input_shape=(window_size, n_features),
# 			kernel_initializer=he_uniform(seed=1),
# 			bias_initializer=he_uniform(seed=11)))
# model.add(Conv1D(10,
# 			(int)(window_size - kernel_size + 1),
# 			kernel_initializer=he_uniform(seed=25),
# 			bias_initializer=he_uniform(seed=91)))
# model.add(Activation('sigmoid'))
# model.add(Flatten())
# model.add(Dense(4, 'tanh',
# 			kernel_initializer=he_uniform(seed=74),
# 			bias_initializer=he_uniform(seed=52)))
# model.compile(optimizer=optimizer, loss=loss)
# model.summary()
# early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)
# model_hist = model.fit(X_train, Y_train, verbose=1, validation_split=0.2, 
# 	shuffle=True, callbacks= [early_stopping_callback], epochs=n_epochs, batch_size=batch_size)
# Y_predict = model.predict(X_test)
# rmse = custom_rmse(Y_predict, Y_test)
# print(np.mean(rmse))
# model.save('trained_model')

model = load_model('trained_model')
Y_predict = model.predict(X_test)
rmse = custom_rmse(Y_predict, Y_test)
print(np.mean(rmse))
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