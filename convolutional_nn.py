import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Activation, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.metrics import RootMeanSquaredError
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from data_processing import cnn_extract_features

trials = np.arange(1, 11)

def cnn_load_data(trial_nums, window_size):
    X, y = [], []
    for trial_num in trial_nums:
        filename_X = f'features/cnn_trial{trial_num}_winsize{window_size}_X.npy'
        filename_y = f'features/cnn_trial{trial_num}_winsize{window_size}_y.npy'
        X.append(np.load(filename_X))
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
    conv_kernel = 10
    model_dep = Sequential()
    model_dep.add(Conv1D(10, conv_kernel, input_shape=(winsize, 10), trainable=False))
    model_dep.add(Conv1D(10, winsize - conv_kernel + 1, trainable=False))
    model_dep.add(Activation('relu'))
    model_dep.add(Flatten())
    model_dep.add(Dense(4, activation='tanh'))
    model_dep.compile(loss='mean_squared_error',
                    optimizer='adam', metrics=RootMeanSquaredError())
    return model_dep

def train_cnn(data_list, window_sizes, optimizers):
    '''
    Params: lists
    Returns: list of errors for each combination of params
    Stores the predictions for each parameter and each trial, as well as an
    error file to the ../predictions folder
    '''
    errors = np.array([])
    for window_size in window_sizes:
        for ix, optimizer in enumerate(optimizers):
            loss_per_trial = np.array([])
            for test_trial_num in trials:
                model = cnn_create_model(window_size)
                data = cnn_extract_features(data_list, window_size, test_trial_num)
                model.compile(loss='mse', optimizer=optimizer,
                                metrics=RootMeanSquaredError())
                # early stopping
                early_stopping_callback = EarlyStopping(
                    monitor='val_loss', 
                    min_delta=0, 
                    patience=10, 
                    verbose=0)
                
                tensorboard_callback = TensorBoard(log_dir='./vis',
                                                profile_batch=0, histogram_freq=1)

                history = model.fit(data['X_train'], data['y_train'], epochs=100, batch_size=128, verbose=0, validation_data=(data['X_test'], data['y_test']), shuffle=True, callbacks= [early_stopping_callback, tensorboard_callback])
              
                plot_learning_curve(
                    history, test_trial_num, window_size)
                plt.show()
                y_preds = model.predict(data['X_test'])
                left_rmse, right_rmse = custom_rmse(data['y_test'], y_preds)
                loss_per_trial = np.append(loss_per_trial, np.mean((left_rmse, right_rmse)))
                clear_session()
            print(f'Loss in each trial: {loss_per_trial}')
            loss_mean = np.mean(loss_per_trial)
            errors = np.append(errors, loss_mean)
            print('Window Size: {} | RMSE: {:.2f}%'.format(
                window_size, loss_mean))
        # model.save('test_model_save_2')
    np.savetxt('err.txt', errors)

def train_rnn():
#     save_model_string = "../Model Checkpoints/" + testing_subject + "LSTM_independent_best_model.hdf5"
    data = cnn_train_test_split(2, 40)
    print(data['X_train'].shape)
    print(data['y_train'].shape)
#     model_checkpoint_callback = ModelCheckpoint(save_model_string, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1)
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)
    model = Sequential()
    model.add(LSTM(30, input_shape=(data['X_train'].shape[1], data['X_train'].shape[-1]), return_sequences = False, activation='relu'))
    model.add(Flatten())
    model.add(Dense(4, activation='tanh'))
    # model = multi_gpu_model(model, gpus=2)
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    model.summary()
    model.fit(data['X_train'], data['y_train'], epochs=5, batch_size=128, verbose=0, validation_split=0.2, shuffle=True, callbacks= [early_stopping_callback])
    model.save('test_rnn_model')
    results = model.evaluate(data['X_test'], data['y_test'])
        
    predictions = model.predict(data['X_test'])
    _,_, gp = custom_rmse(data['y_test'], predictions)
    
    plt.figure(1)
    plt.plot(gp['left_true'])
    plt.plot(gp['left_pred'])
    plt.legend(['GT', 'Pred'])
    plt.show()
    return model
    
def custom_rmse(y_true, y_pred):
    #Raw values and Prediction are in X,Y
    labels, theta, gp = {}, {}, {}

    #Separate legs
    left_true = y_true[:, :2]
    right_true = y_true[:, 2:]
    left_pred = y_pred[:, :2]
    right_pred = y_pred[:, 2:]
    
    #Calculate cosine distance
    left_num = np.sum(np.multiply(left_true, left_pred), axis=1)
    left_denom = np.linalg.norm(left_true, axis=1) * np.linalg.norm(left_pred, axis=1)
    right_num = np.sum(np.multiply(right_true, right_pred), axis=1)
    right_denom = np.linalg.norm(right_true, axis=1) * np.linalg.norm(right_pred, axis=1)

    left_cos = left_num / left_denom
    right_cos = right_num / right_denom
    
    #Clip large values and small values
    left_cos = np.minimum(left_cos, np.zeros(left_cos.shape)+1)
    left_cos = np.maximum(left_cos, np.zeros(left_cos.shape)-1)
    
    right_cos = np.minimum(right_cos, np.zeros(right_cos.shape)+1)
    right_cos = np.maximum(right_cos, np.zeros(right_cos.shape)-1)
    
    #Get theta error
    left_theta = np.arccos(left_cos)
    right_theta = np.arccos(right_cos)
    
    #Get gait phase error
    left_gp_error = left_theta * 100 / (2*np.pi)
    right_gp_error = right_theta * 100 / (2*np.pi)
    
    #Get rmse
    left_rmse = np.sqrt(np.mean(np.square(left_gp_error)))
    right_rmse = np.sqrt(np.mean(np.square(right_gp_error)))

    return left_rmse, right_rmse


def plot_err(param):
    # Plot err against the parameter
    errors = np.loadtxt('err.txt')
    plt.plot(param, errors)
    plt.xticks(param)
    plt.xlabel('window size (ms)')
    plt.ylabel('rmse (%)')

    # zip joins x and y coordinates in pairs
    for x, y in zip(param, errors):

        label = "{:.2f}".format(y)

        plt.annotate(label,  # this is the text
            (x, y),  # this is the point to label
            textcoords="offset points",  # how to position the text
            xytext=(0, -10),  # distance from text to points (x,y)
            ha='center')  # horizontal alignment can be left, right or center
    plt.title('Convolutional Neural Network')
    plt.grid()
    plt.show()

def plot_gait_phase(gp_true, gp_pred):
    plt.plot(gp_true)
    plt.plot(gp_pred)
    plt.title('CNN Gait Phase')
    plt.ylabel('Gait Phase (%)')
    plt.legend(['True', 'Pred'])

def plot_learning_curve(history, trial, winsize):
    plt.plot(history.history['root_mean_squared_error'],
             label='RMSE (training data)')
    plt.plot(history.history['val_root_mean_squared_error'],
            label='RMSE (validation data)')
    plt.title(f'RMSE for trial {trial} window size {winsize}')
    plt.ylabel('RMSE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
