import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

trials = np.arange(1, 11)

def cnn_load_data(trial_nums, window_size):
    X, y = [], []
    for trial_num in trial_nums:
        filename_X = f'features/cnn_trial{trial_num}_winsize{window_size}_X.npy'
        filename_y = f'features/cnn_trial{trial_num}_winsize{window_size}_y.npy'
        X.append(np.load(filename_X).squeeze())
        y.append(np.load(filename_y))
    X = np.concatenate(X, axis=0)
    y = np.concatenate(y, axis=0)
    return X, y


def cnn_train_test_split(test_trial_num, window_size):
    label_columns = np.arange(10, 14)
    # Leave the test trial and put the other 4 in a training set
    # Split up both features(X) and labels(y) as training sets and test sets
    train_trials = np.delete(trials, test_trial_num-1)
    test_trial = [test_trial_num]
    X_train, y_train = cnn_load_data(train_trials, window_size)
    X_test, y_test = cnn_load_data(test_trial, window_size)
    data = {'X_train': X_train, 'y_train': y_train,
            'X_test': X_test, 'y_test': y_test}
    return data


def preprocess_data(X_train, X_test):
    # Feature Scaling
    scaler = StandardScaler()
    X_train, X_test = scaler.fit_transform(
        X_train), scaler.fit_transform(X_test)
    return X_train, X_test

def cnn_create_model(winsize):
    # create model
    model = Sequential()
    kersize = 20
    # add model layers
    model.add(Conv1D(filters=10, input_shape=(winsize, 10), kernel_size=kersize))
    model.add(Conv1D(filters=10, kernel_size=winsize-kersize+1, activation="relu"))
    model.add(Flatten())
    model.add(Dense(4, activation="tanh"))
    return model

def train_cnn(window_sizes, num_layers, num_nodes, optimizers):
    '''
    Params: lists
    Returns: list of errors for each combination of params
    Stores the predictions for each parameter and each trial, as well as an
    error file to the ../predictions folder
    '''
    errors = []
    for window_size in window_sizes:
        for num_layer in num_layers:
            for num_node in num_nodes:
                model = cnn_create_model(window_size)
                for ix, optimizer in enumerate(optimizers):
                    model.compile(loss='mse', optimizer=optimizer)
                    loss_per_trial = []
                    accuracy = []
                    for test_trial_num in trials:
                        data = cnn_train_test_split(test_trial_num, window_size)
                        model.fit(x=data['X_train'], y=data['y_train'],
                                  epochs=1, batch_size=128, verbose=0)
                    for test_trial_num in trials:
                        data = cnn_train_test_split(test_trial_num, window_size)
                        y_preds = model.predict(data['X_test'])
                        loss_per_trial = np.mean(custom_rmse(data['y_test'], y_preds))
                    loss_mean = np.mean(loss_per_trial)
                    errors.append(loss_mean)
                    print('Window Size: {} \nRSME: {:.2f}'.format(
                        window_size, loss_mean))
                # model.save('test_model_save_2')
#     errs = errors.to_numpy()
#     np.save('predictions/err.txt', errs)
#     return errs
    return

def custom_rmse(y_true, y_pred):
    #Raw values and Prediction are in X,Y

    #Separate legs
    left_true = y_true[:, :2]
    right_true = y_true[:, 2:]
    left_pred = y_pred[:, :2]
    right_pred = y_pred[:, 2:]

    #Convert to polar
    left_true_theta = np.arctan2(left_true[:, 1], left_true[:, 0])
    right_true_theta = np.arctan2(right_true[:, 1], right_true[:, 0])
    left_pred_theta = np.arctan2(left_pred[:, 1], left_pred[:, 0])
    right_pred_theta = np.arctan2(right_pred[:, 1], right_pred[:, 0])

    #Bring into range of 0 to 2pi
    left_true_theta = np.mod(left_true_theta + 2*np.pi, 2*np.pi)
    right_true_theta = np.mod(right_true_theta + 2*np.pi, 2*np.pi)
    left_pred_theta = np.mod(left_pred_theta + 2*np.pi, 2*np.pi)
    right_pred_theta = np.mod(right_pred_theta + 2*np.pi, 2*np.pi)

    #Interpolate from 0 to 100%
    left_gp_true = 100*left_true_theta / (2*np.pi)
    right_gp_true = 100*right_true_theta / (2*np.pi)
    left_gp_pred = 100*left_pred_theta / (2*np.pi)
    right_gp_pred = 100*right_pred_theta / (2*np.pi)

    #Diff
    left_diff = np.subtract(left_gp_true, left_gp_pred)
    right_diff = np.subtract(right_gp_true, right_gp_pred)
    left_rmse = np.sqrt(np.mean(np.square(left_diff)))
    right_rmse = np.sqrt(np.mean(np.square(right_diff)))

    return left_rmse, right_rmse

def plot_err(param):
    # Plot err against the parameter
    errors = np.loadtxt('predictions/err')
    plt.plot(param, error)
    plt.xticks(param)
    plt.xlabel('window size (ms)')
    plt.ylabel('mean absolute error')

    # zip joins x and y coordinates in pairs
    for x, y in zip(param, accuracy):

        label = "{:.2f}".format(y)

        plt.annotate(label,  # this is the text
            (x, y),  # this is the point to label
            textcoords="offset points",  # how to position the text
            xytext=(0, -10),  # distance from text to points (x,y)
            ha='center')  # horizontal alignment can be left, right or center
    plt.title('Convolutional Neural Network')
    plt.grid()
    plt.show()
