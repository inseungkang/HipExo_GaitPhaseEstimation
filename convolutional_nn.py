import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Flatten, Activation, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
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
    kersize2 = int(0.5*winsize - kersize + 1)
    # add model layers
    model.add(Conv1D(filters=10, input_shape=(winsize, 10), kernel_size=kersize))
    model.add(Conv1D(filters=10, kernel_size=10))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(4, activation="tanh"))
    model.summary()
    return model

def train_cnn(window_sizes, num_layers, num_nodes, optimizers):
    '''
    Params: lists
    Returns: list of errors for each combination of params
    Stores the predictions for each parameter and each trial, as well as an
    error file to the ../predictions folder
    '''
    errors = np.array([])
    for window_size in window_sizes:
        for num_layer in num_layers:
            for num_node in num_nodes:
                for ix, optimizer in enumerate(optimizers):
                    loss_per_trial = np.array([])
                    for test_trial_num in trials:
                        model = cnn_create_model(window_size)
                        data = cnn_train_test_split(test_trial_num, window_size)
                        model.compile(loss='mse', optimizer=optimizer)
                        # early stopping
                        early_stopping_callback = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=5, 
                            verbose=0)
                        
                        tensorboard_callback = TensorBoard(log_dir='./vis',
                                                      profile_batch=0, histogram_freq=1)

                        model.fit(
                            data['X_train'], 
                            data['y_train'], 
                            epochs=3, 
                            batch_size=128, 
                            verbose=1,
                            shuffle=True, 
                            validation_split=0.15,
                            callbacks=[early_stopping_callback, tensorboard_callback])
                            
                        y_preds = model.predict(data['X_test'])
                        loss_per_trial = np.append(loss_per_trial, np.mean(custom_rmse(data['y_test'], y_preds)))
                        clear_session()
                    print(f'Loss in each trial: {loss_per_trial}')
                    loss_mean = np.mean(loss_per_trial)
                    errors = np.append(errors, loss_mean)
                    print('Window Size: {} | RMSE: {:.2f}%'.format(
                        window_size, loss_mean))
                model.save('test_model_save_2')
    np.savetxt('err.txt', errors)

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

    #Separate legs
    labels['left_true'] = left_true
    labels['right_true'] = right_true
    labels['left_pred'] = left_pred
    labels['right_pred'] = right_pred

    for key, value in labels.items(): 
        #Convert to polar
        theta[key] = np.arctan2(value[:, 1], value[:, 0])
        
        #Bring into range of 0 to 2pi
        theta[key] = np.mod(theta[key] + 2*np.pi, 2*np.pi)

        #Interpolate from 0 to 100%
        gp[key] = 100*theta[key] / (2*np.pi)

    return left_rmse, right_rmse, gp




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

def plot_gait_phase(y_true, y_pred):
    plt.figure()
    plt.plot(y_true)
    plt.plot(y_pred)
    plt.title('CNN Gait Phase')
    plt.ylabel('Gait Phase (%)')
