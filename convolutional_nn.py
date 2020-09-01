import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from sklearn.model_selection import train_test_split

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

def cnn_create_model(winsize):
    # create model
    model = Sequential()
    
    # add model layers
    model.add(Conv2D(64, input_shape=(winsize, 10, 1), kernel_size=(3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(4))

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
                print(model.summary())
                for ix, optimizer in enumerate(optimizers):
                    model.compile(loss='mae', optimizer=optimizer)
                    loss_per_trial = []
                    for test_trial_num in trials:
                        data = cnn_train_test_split(test_trial_num, window_size)
                        model.fit(x=data['X_train'], y=data['y_train'],
                                  epochs=1, batch_size=128, verbose=0)
#                         loss_per_trial.append(model.evaluate(
#                                 data['X_test'], data['y_test']))
#                         y_preds = model.predict(data['X_test'])
                        # writes the predictions to a file in predictions file
                        # file_name = f'predictions/{mode}_wsize{window_size}_{num_layer}layers_{num_node}nodes_optimizer{ix+1}_trial{test_trial_num}.txt'
                        # y_preds.to_csv(file_name, index=False)
                    for test_trial_num in trials:
                        data = cnn_train_test_split(test_trial_num, window_size)
                        loss_per_trial.append(model.evaluate(
                                data['X_test'], data['y_test']))
                        y_preds = model.predict(data['X_test'])
                    loss_mean = np.mean(loss_per_trial) * 100
                    errors.append(loss_mean)
                    print('Window Size: {} \nAccuracy: {:.2f}%'.format(
                        window_size, loss_mean))
                model.save('test_model_save')
#     errs = errors.to_numpy()
#     np.save('predictions/err.txt', errs)
#     return errs
    return
    


