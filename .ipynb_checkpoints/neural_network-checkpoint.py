import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD, Adam, RMSprop
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy

SEED = 42
trials = np.arange(1, 6)
label_columns = ['leftGaitPhaseX', 'leftGaitPhaseY',
                 'rightGaitPhaseX', 'rightGaitPhaseY']

def load_data(trial_nums, window_size):
    """ return concatenated data from trials in tiral_nums"""
    data = pd.DataFrame()
    for trial_num in trial_nums:
        file_name = f'features/trial{trial_num}_winsize{window_size}.txt'
        data = data.append(pd.read_csv(file_name))
    return data


def create_model(out_dim, num_layer, num_node):
    # create a linera regression model
    model = Sequential()
    for x in range(num_layer):
        model.add(Dense(num_node, activation="relu"))
    model.add(Dense(out_dim))
    return model

def create_model_class(num_layer, num_node):
    # create a classification model
    model = Sequential()
    for x in range(num_layer):
        model.add(Dense(num_node, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    return model

def train_test_split(test_trial_num, window_size, label_columns):
    # Leave the test trial and put the other 4 in a training set
    # Splits the features and the label for both training set and test set
    train_trials = np.delete(trials, test_trial_num-1)
    test_trial = [test_trial_num]
    train_set = load_data(train_trials, window_size)
    test_set = load_data(test_trial, window_size)

    # Split the features and the target
    X_train = train_set.drop(columns=label_columns).to_numpy()
    y_train = train_set[label_columns].to_numpy()

    X_test = test_set.drop(columns=label_columns).to_numpy()
    y_test = test_set[label_columns].to_numpy()

    X_train, X_test = preprocess_data(X_train, X_test)

    data = {'X_train':X_train, 'y_train':y_train, 'X_test':X_test, 'y_test':y_test}
    return data

def train_test_split_uni(test_trial_num, window_size):
    # Leave the test trial and put the other 4 in a training set
    # Splits the features and the label for both training set and test set, and for both right side and left side
    train_trials = np.delete(trials, test_trial_num-1)
    test_trial = [test_trial_num]
    train_set = load_data(train_trials, window_size)
    test_set = load_data(test_trial, window_size)

    train_left = train_set[train_set.columns.drop(list(train_set.filter(regex='right')))]
    test_left = test_set[test_set.columns.drop(list(test_set.filter(regex='right')))]
    X_train_left = train_left[train_left.columns.drop(list(train_left.filter(regex='Gait')))].to_numpy()
    y_train_left = train_left.filter(regex='Gait').to_numpy()
    X_test_left = test_left[test_left.columns.drop(list(test_left.filter(regex='Gait')))].to_numpy()
    y_test_left = test_left.filter(regex='Gait').to_numpy()

    
    train_right = train_set[train_set.columns.drop(list(train_set.filter(regex='left')))]
    test_right = test_set[test_set.columns.drop(list(test_set.filter(regex='left')))]   
    X_train_right = train_right[train_right.columns.drop(list(train_right.filter(regex='Gait')))].to_numpy()
    y_train_right = train_right.filter(regex='Gait').to_numpy()
    X_test_right = test_right[test_right.columns.drop(list(test_right.filter(regex='Gait')))].to_numpy()
    y_test_right = test_right.filter(regex='Gait').to_numpy()

    X_train_left, X_test_left = preprocess_data(X_train_left, X_test_left)
    X_train_right, X_test_right = preprocess_data(X_train_right, X_test_right)

    data = {'left': {'X_train': X_train_left, 'y_train': y_train_left, 'X_test': X_test_left, 'y_test': y_test_left}, 'right': {'X_train': X_train_right, 'y_train': y_train_right, 'X_test': X_test_right, 'y_test': y_test_right}}

    return data


def preprocess_data(X_train, X_test):
    # Feature Scaling
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(X_train), scaler.fit_transform(X_test)
    return X_train, X_test


def train_nn(window_sizes, num_layers, num_nodes, optimizers, mode='bi'):
    '''
    All parameters except for mode should be in an array/list format
    mode should be one of following: ['bi', 'left', 'right', 'classification']
    Returns an array of errors for each parameter
    Stores the predictions for each parameter and each trial, as well as an
    error file to the ../predictions folder
    '''
    errors = []
    for window_size in window_sizes:
        for num_layer in num_layers:
            for num_node in num_nodes:
                if mode == 'bi':
                    model = create_model(4, num_layer, num_node)
                if mode == 'classification':
                    model = create_model_class(num_layer, num_node)
                else:
                    model = create_model(2, num_layer, num_node)
                for ix, optimizer in enumerate(optimizers):
                    if mode == 'classification':
                        model.compile(
                            loss='binary_crossentropy', optimizer=optimizer, metrics=[keras.metrics.BinaryAccuracy()])
                    else:
                        model.compile(loss='mae', optimizer=optimizer)
                    loss_per_trial = []
                    for test_trial_num in trials:
                        if mode == 'bi':
                            data = train_test_split(test_trial_num, window_size, label_columns)
                        elif mode == 'classification':
                            data = train_test_split(
                                test_trial_num, window_size, ['y'])
                        else:
                            data = train_test_split_uni(test_trial_num, window_size)
                            data = data[mode]
                        model.fit(x=data['X_train'], y=data['y_train'], epochs=5, batch_size=128, verbose=2)
                        if mode == 'classification': 
                            _, accuracy = model.evaluate(data['X_test'], data['y_test'])
                            loss_per_trial.append(accuracy)
                            y_preds = pd.DataFrame(
                                model.predict_classes(data['X_test']))
                        else:
                            loss_per_trial.append(model.evaluate(data['X_test'], data['y_test']))
                            y_preds = pd.DataFrame(
                                model.predict(data['X_test']))
                        # writes the predictions to a file in predictions file
                        file_name = f'predictions/{mode}_wsize{window_size}_{num_layer}layers_{num_node}nodes_optimizer{ix+1}_trial{test_trial_num}.txt'
                        y_preds.to_csv(file_name, index=False)
                    loss_mean = np.mean(loss_per_trial) * 100
                    errors.append(loss_mean)
                    print('Mode: {}\nWindow Size: {} | {} Hidden Layers | {} nodes\nOptimizer: {}\nAccuracy: {:.2f}%'.format(mode, window_size, num_layer, num_node, ix+1, loss_mean))
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv('predictions/err.txt', index=False)
    return np.array(errors)


def plot_err(param):
    # Plot err against the parameter
    errors = pd.read_csv('predictions/err.txt').to_numpy()
    accuracy = 100 - errors
    plt.plot(param, accuracy)
    plt.xticks(param)
    plt.xlabel('window size')
    plt.ylabel('accuracy')
    plt.title('Unilateral Model')
    plt.grid()
    plt.show()

def plot_time_series(best_window_size):
    # plot the time series of predicted gait phase vs labeled gait phase using prediction files in ../predictions folder for each trial
    for test_trial_num in trials:
        file_name = f'predictions/bi_wsize40_3layers_15nodes_optimizer1_trial{test_trial_num}.txt'
        
        data = train_test_split(test_trial_num, best_window_size[0])
        label = data["y_test"]
        theta = np.arccos(label[:,(0,2)])
        theta = np.where(label[:,(1,3)]>=0, theta, 2*np.pi-theta)
        gp = theta/(2*np.pi) * 100

        pred = pd.read_csv(file_name).to_numpy()
        pred = np.where(pred<=1, pred, 1)
        pred = np.where(pred>=-1, pred, -1)
        theta_pred = np.arccos(pred[:,(0,2)])
        theta_pred = np.where(pred[:,(1,3)]>=0, theta_pred, 2*np.pi-theta_pred)
        gp_pred = theta_pred/(2*np.pi) * 100

        plt.figure(figsize=(13,8))
        plt.subplot(211)
        plt.title('Left')
        plt.plot(gp[:,0], label='label')
        plt.plot(gp_pred[:,0], 'red', alpha=0.8, label='predictions')
        plt.legend(loc='center left')

        plt.subplot(212)
        plt.title('Right')
        plt.plot(gp[:,1], label='label')
        plt.plot(gp_pred[:,1], 'red', alpha=0.8, label='predictions')
        plt.legend(loc='center left')

    plt.show()
