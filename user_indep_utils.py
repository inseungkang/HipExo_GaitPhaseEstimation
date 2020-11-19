import itertools
import numpy as np
import pandas as pd
from tensorflow.initializers import he_uniform
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad
from tensorflow.keras.backend import clear_session
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Activation, Flatten, LSTM, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def get_model_configs_independent(hyperparam_space):
    model_configs = []
    model_type = hyperparam_space['model']
    for window_size in hyperparam_space['window_size']:
        if (model_type != 'mlp'):
            type_specific_params = list(hyperparam_space[model_type].keys())

            type_specific_possibilities = []

            for param in type_specific_params:
                type_specific_possibilities.append(hyperparam_space[model_type][param])

            type_specific_config_tuples = itertools.product(*type_specific_possibilities)
            type_specific_configs = []
            for config in type_specific_config_tuples:
                type_specific_config = {}
                for i, value in enumerate(config):
                    type_specific_config[type_specific_params[i]] = value
                type_specific_configs.append(type_specific_config)

        dense_params = list(hyperparam_space['dense'].keys())

        dense_possibilities = []

        for param in dense_params:
            dense_possibilities.append(hyperparam_space['dense'][param])

        dense_config_tuples = itertools.product(*dense_possibilities)
        dense_configs = []
        for config in dense_config_tuples:
            dense_config = {}
            for i, value in enumerate(config):
                dense_config[dense_params[i]] = value
            dense_configs.append(dense_config)

        optim_params = list(hyperparam_space['optimizer'].keys())

        optim_possibilities = []

        for param in optim_params:
            optim_possibilities.append(hyperparam_space['optimizer'][param])
        optim_config_tuples = itertools.product(*optim_possibilities)
        optim_configs = []
        for config in optim_config_tuples:
            optim_config = {}
            for i, value in enumerate(config):
                optim_config[optim_params[i]] = value
            optim_configs.append(optim_config)

        training_params = list(hyperparam_space['training'].keys())

        training_possibilities = []

        for param in training_params:
            training_possibilities.append(hyperparam_space['training'][param])

        training_config_tuples = itertools.product(*training_possibilities)
        training_configs = []
        for config in training_config_tuples:
            training_config = {}
            for i, value in enumerate(config):
                training_config[training_params[i]] = value
            training_configs.append(training_config)
        
        if model_type != 'mlp':
            possible_configs = itertools.product(type_specific_configs, dense_configs, optim_configs, training_configs)
            config_count = 0
            for config in possible_configs:
                config_count += 1
                config_obj = {
                    'window_size': window_size,
                    'model': model_type,
                    'dense': config[1],
                    'optimizer': config[2],
                    'training': config[3]
                }
                config_obj[model_type] = config[0]
                model_configs.append(config_obj)
        else:
            possible_configs = itertools.product(dense_configs, optim_configs, training_configs)
            config_count = 0
            for config in possible_configs:
                config_count += 1
                config_obj = {
                    'window_size': window_size,
                    'model': model_type,
                    'dense': config[0],
                    'optimizer': config[1],
                    'training': config[2]
                }
                model_configs.append(config_obj)

    return model_configs

def cnn_extract_features_independent(data_list, window_size, test_subject):
    testing_data = []
    training_data = []
    for subject in data_list.keys():
        if subject == test_subject:
            testing_data.append(data_list[subject])
        else:
            training_data.append(data_list[subject])
       
    X_test = np.zeros((1, window_size, 12))
    Y_test = np.zeros((1, 4))
    X_train = np.zeros((1, window_size, 12))
    Y_train = np.zeros((1, 4))
    data_out = {}
    
    # Generate testing data
    for i, data in enumerate(testing_data):
        data = data.to_numpy()
        
        trial_X = data[:, :-4]
        trial_Y = data[:, -4:]
        
        #Sliding window
        shape_des = (trial_X.shape[0] - window_size +
                    1, window_size, trial_X.shape[-1])
        strides_des = (
            trial_X.strides[0], trial_X.strides[0], trial_X.strides[1])
        trial_X = np.lib.stride_tricks.as_strided(trial_X, shape=shape_des,
                                                strides=strides_des)
            
        trial_Y = trial_Y[window_size-1:]

        X_test = np.concatenate([X_test, trial_X], axis=0)
        Y_test = np.concatenate([Y_test, trial_Y], axis=0)
    
    X_test = X_test[1:, :, :]
    Y_test = Y_test[1:, :]

    data_out['X_test'] = X_test
    data_out['y_test'] = Y_test
    
    # Generate Training Data
    for i, data in enumerate(training_data):       
        data = data.to_numpy()
        
        trial_X = data[:, :-4]
        trial_Y = data[:, -4:]

        #Sliding window
        shape_des = (trial_X.shape[0] - window_size +
                        1, window_size, trial_X.shape[-1])
        strides_des = (
            trial_X.strides[0], trial_X.strides[0], trial_X.strides[1])
        trial_X = np.lib.stride_tricks.as_strided(trial_X, shape=shape_des,
                                                    strides=strides_des)
        trial_Y = trial_Y[window_size-1:]

        X_train = np.concatenate([X_train, trial_X], axis=0)
        Y_train = np.concatenate([Y_train, trial_Y], axis=0)

    X_train = X_train[1:, :, :]
    Y_train = Y_train[1:, :]
    data_out['X_train'] = X_train
    data_out['y_train'] = Y_train
    return data_out

def deploy_cnn_extract_features_independent(data_list, window_size):
    testing_data = []
    training_data = []
    for subject in data_list.keys():
        training_data.append(data_list[subject])
       
    X_train = np.zeros((1, window_size, 10))
    Y_train = np.zeros((1, 4))
    data_out = {}
    
    # Generate Training Data
    for i, data in enumerate(training_data):       
        data = data.to_numpy()
        
        trial_X = data[:, :-4]
        trial_Y = data[:, -4:]

        #Sliding window
        shape_des = (trial_X.shape[0] - window_size +
                        1, window_size, trial_X.shape[-1])
        strides_des = (
            trial_X.strides[0], trial_X.strides[0], trial_X.strides[1])
        trial_X = np.lib.stride_tricks.as_strided(trial_X, shape=shape_des,
                                                    strides=strides_des)
        trial_Y = trial_Y[window_size-1:]

        X_train = np.concatenate([X_train, trial_X], axis=0)
        Y_train = np.concatenate([Y_train, trial_Y], axis=0)

    X_train = X_train[1:, :, :]
    Y_train = Y_train[1:, :]
    data_out['X_train'] = X_train
    data_out['y_train'] = Y_train
    return data_out

def get_dataset_independent(model_type, data_list, window_size, test_subject):
    if model_type == 'cnn':
        return cnn_extract_features_independent(data_list, window_size, test_subject)
    else:
        raise Exception('No dataset for model type')

def results_mapper_indep(x):
    out = {}
    out['window_size'] = x['model_config']['window_size']
    model_type = x['model_config']['model']
    out['model_type'] = model_type
    if model_type != 'mlp':
        for key in x['model_config'][model_type].keys():
            out['{}_{}'.format(model_type, key)] = x['model_config'][model_type][key]

    for key in x['model_config']['dense'].keys():
        out['dense_{}'.format(key)] = x['model_config']['dense'][key]

    for key in x['model_config']['optimizer'].keys():
        out['optim_{}'.format(key)] = x['model_config']['optimizer'][key]
        
    for key in x['model_config']['training'].keys():
        out['training_{}'.format(key)] = x['model_config']['training'][key]
    
    out['left_rmse_mean'] = x['left_rmse_mean']
    out['right_rmse_mean'] = x['right_rmse_mean']
    return out

# Creates an LSTM model based on the specified configuration
def lstm_model(sequence_length, n_features, lstm_config, dense_config, optim_config, X_train):
    model = Sequential()
    # norm_layer = Normalization(input_shape=(sequence_length, n_features))
    # norm_layer.adapt(X_train)
    # model.add(norm_layer)
    model.add(BatchNormalization(input_shape=(sequence_length, n_features)))
    model.add(LSTM(
                return_sequences = False,
                kernel_initializer=he_uniform(seed=1),
                recurrent_initializer=he_uniform(seed=11),
                bias_initializer=he_uniform(seed=25),
                **lstm_config))
    model.add(Dense(4,
                kernel_initializer=he_uniform(seed=91),
                bias_initializer=he_uniform(seed=74),
                **dense_config))
    model.compile(**optim_config)
    return model

# Creates a CNN model based on the specified configuration
def cnn_model(window_size, n_features, cnn_config, dense_config, optim_config, X_train):
    conv_kernel = cnn_config['kernel_size']
    model = Sequential()
    # norm_layer = Normalization(input_shape=(window_size, n_features))
    # norm_layer.adapt(X_train)
    # model.add(norm_layer)
    model.add(BatchNormalization(input_shape=(window_size, n_features)))
    model.add(Conv1D(10,
                conv_kernel,
                input_shape=(window_size, n_features),
                kernel_initializer=he_uniform(seed=1),
                bias_initializer=he_uniform(seed=11)))
                # kernel_initializer=HeUniform(seed=1),
                # bias_initializer=HeUniform(seed=11)))
    model.add(Conv1D(10,
                (int)(window_size - conv_kernel + 1),
                kernel_initializer=he_uniform(seed=25),
                bias_initializer=he_uniform(seed=91)))
                # kernel_initializer=HeUniform(seed=25),
                # bias_initializer=HeUniform(seed=91)))
    model.add(Activation(cnn_config['activation']))
    model.add(Flatten())
    model.add(Dense(4,
                dense_config['activation'],
                kernel_initializer=he_uniform(seed=74),
                bias_initializer=he_uniform(seed=52)))
                # kernel_initializer=HeUniform(seed=74),
                # bias_initializer=HeUniform(seed=52)))
    model.compile(**optim_config)
    return model

# Creates a deeper CNN model based on the specified configuration
def big_cnn_model(window_size, n_features, cnn_config, dense_config, optim_config, X_train):
    print(cnn_config)
    conv_kernel = cnn_config['kernel_size']
    layers = cnn_config['layers'] - 2
    output_kernel = window_size - conv_kernel + 1
    
    model = Sequential()
    
    model.add(BatchNormalization(input_shape=(window_size, n_features)))
    model.add(Conv1D(10,
                conv_kernel,
                input_shape=(window_size, n_features),
                kernel_initializer=he_uniform(seed=1),
                bias_initializer=he_uniform(seed=11)))
    
    kernel_factor = 0.75
    while layers:
        kernel_size = (int)(np.ceil(kernel_factor * conv_kernel))
        output_kernel -= kernel_size - 1
        model.add(Conv1D(10,
                kernel_size,
                input_shape=(window_size, n_features),
                kernel_initializer=he_uniform(seed=1),
                bias_initializer=he_uniform(seed=11)))
        layers -= 1
        kernel_factor -= 0.25
    
    output_kernel -= cnn_config['output_size'] - 1
    model.add(Conv1D(10, (int)(output_kernel),
                kernel_initializer=he_uniform(seed=25),
                bias_initializer=he_uniform(seed=91)))
    model.add(Activation(cnn_config['activation']))
    model.add(Flatten())
    
    dense_layers = dense_config['layers'][0]
    dense_nodes = dense_config['layers'][1]
    while dense_layers:
        model.add(Dense(dense_nodes,
                    dense_config['activation'],
                    kernel_initializer=he_uniform(seed=74),
                    bias_initializer=he_uniform(seed=52)))
        dense_layers -= 1
    model.add(Dense(4,
                dense_config['activation'],
                kernel_initializer=he_uniform(seed=74),
                bias_initializer=he_uniform(seed=52)))
    model.compile(**optim_config, metrics=['accuracy'])
    return model


# Creates an MLP model based on the specified configuration
def mlp_model(n_features, dense_config, optim_config, X_train):
    model = Sequential()
    # norm_layer = Normalization(input_shape=(n_features,))
    # norm_layer.adapt(X_train)
    # model.add(norm_layer)
    model.add(BatchNormalization(input_shape=(n_features,)))
    for x in range(dense_config['num_layers']):
        model.add(Dense(dense_config['num_nodes'], activation=dense_config['activation']))
    model.add(Dense(4))
    model.compile(**optim_config)
    return model

# Uses cosine distance to calculate the RMSE
#  in gait phase percentage
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
    
    # What if denominator is zero (model predicts 0 for both X and Y)
    left_cos[np.isnan(left_cos)] = 0
    right_cos[np.isnan(right_cos)] = 0
    
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

    return left_rmse, right_rmse

# Creates the appropriate model based on the configuration
def create_model_subject(model_config, dataset):
    if 'lr' in model_config['optimizer']:
        lr = model_config['optimizer']['lr']
        optim_config = {k:v for (k, v) in model_config['optimizer'].items() if k != 'lr'}
        if optim_config['optimizer'] == 'adam':
            optim_config['optimizer'] = Adam(learning_rate=lr)
        if optim_config['optimizer'] == 'sgd':
            optim_config['optimizer'] = SGD(learning_rate=lr)
        if optim_config['optimizer'] == 'adagrad':
            optim_config['optimizer'] = Adagrad(learning_rate=lr)
        if optim_config['optimizer'] == 'rmsprop':
            optim_config['optimizer'] = RMSprop(learning_rate=lr)
    else:
        optim_config = model_config['optimizer']
    if (model_config['model'] == 'lstm'):
        return lstm_model(sequence_length=model_config['window_size'],
                          n_features=10, 
                           lstm_config=model_config['lstm'],
                           dense_config=model_config['dense'],
                           optim_config=optim_config,
                           X_train=dataset['X_train'].squeeze())
    elif (model_config['model'] == 'cnn'):
        return cnn_model(window_size=model_config['window_size'],
                         n_features=10,
                         cnn_config=model_config['cnn'],
                         dense_config=model_config['dense'],
                         optim_config=optim_config,
                         X_train=dataset['X_train'])
    elif (model_config['model'] == 'mlp'):
        return mlp_model(n_features=50,
                         dense_config=model_config['dense'],
                         optim_config=optim_config,
                         X_train=dataset['X_train'])
    else:
        raise Exception('No model generator for model type')

# Creates the appropriate model based on the configuration
def create_big_model_subject(model_config, dataset):
    if 'lr' in model_config['optimizer']:
        lr = model_config['optimizer']['lr']
        optim_config = {k:v for (k, v) in model_config['optimizer'].items() if k != 'lr'}
        if optim_config['optimizer'] == 'adam':
            optim_config['optimizer'] = Adam(learning_rate=lr)
        if optim_config['optimizer'] == 'sgd':
            optim_config['optimizer'] = SGD(learning_rate=lr)
        if optim_config['optimizer'] == 'adagrad':
            optim_config['optimizer'] = Adagrad(learning_rate=lr)
        if optim_config['optimizer'] == 'rmsprop':
            optim_config['optimizer'] = RMSprop(learning_rate=lr)
    else:
        optim_config = model_config['optimizer']
    if (model_config['model'] == 'lstm'):
        return lstm_model(sequence_length=model_config['window_size'],
                          n_features=10, 
                           lstm_config=model_config['lstm'],
                           dense_config=model_config['dense'],
                           optim_config=optim_config,
                           X_train=dataset['X_train'].squeeze())
    elif (model_config['model'] == 'cnn'):
        return big_cnn_model(window_size=model_config['window_size'],
                         n_features=12,
                         cnn_config=model_config['cnn'],
                         dense_config=model_config['dense'],
                         optim_config=optim_config,
                         X_train=dataset['X_train'])
    elif (model_config['model'] == 'mlp'):
        return mlp_model(n_features=50,
                         dense_config=model_config['dense'],
                         optim_config=optim_config,
                         X_train=dataset['X_train'])
    else:
        raise Exception('No model generator for model type')

def train_models_independent(model_type, hyperparameter_configs, data):
    results = []
    for model_config in hyperparameter_configs:
        current_result = {}
        current_result['model_config'] = model_config
        current_result['left_validation_rmse'] = []
        current_result['right_validation_rmse'] = []
        subjects = data.keys()
        for test_subject in subjects:
            dataset = get_dataset_independent(model_type, data, model_config['window_size'], test_subject)
            model = create_model_subject(model_config.copy(), dataset)
            model.summary()
            early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)
            model_hist = model.fit(dataset['X_train'], dataset['y_train'], verbose=1, validation_split=0.2, shuffle=True, callbacks= [early_stopping_callback], **model_config['training'])

            predictions = model.predict(dataset['X_test'])
            left_rmse, right_rmse = custom_rmse(dataset['y_test'], predictions)

            current_result['left_validation_rmse'].append(left_rmse)
            current_result['right_validation_rmse'].append(right_rmse)
            clear_session()
        results.append(current_result)
    
    per_trial_results = []
    for trial in results:
        left_val_rmse = trial['left_validation_rmse']
        right_val_rmse = trial['right_validation_rmse']
        for i, test_subject in enumerate(data.keys()):
            trial_result = {}
            trial_result['test_subject'] = test_subject
            trial_result['window_size'] = trial['model_config']['window_size']

            if (model_type != 'mlp'):
                for key in trial['model_config'][model_type].keys():
                    trial_result['{}_{}'.format(model_type, key)] = trial['model_config'][model_type][key]

            for key in trial['model_config']['dense'].keys():
                trial_result['dense_{}'.format(key)] = trial['model_config']['dense'][key]

            for key in trial['model_config']['optimizer'].keys():
                trial_result['optim_{}'.format(key)] = trial['model_config']['optimizer'][key]

            for key in trial['model_config']['training'].keys():
                trial_result['training_{}'.format(key)] = trial['model_config']['training'][key]
            trial_result['left_validation_rmse'] = left_val_rmse[i-1]
            trial_result['right_validation_rmse'] = right_val_rmse[i-1]
            per_trial_results.append(trial_result)
    df_per_trial_results = pd.DataFrame(per_trial_results)

    for model in results:
        model['left_rmse_mean'] = np.mean(model['left_validation_rmse'])
        model['right_rmse_mean'] = np.mean(model['right_validation_rmse'])
                
    averaged_results = list(map(results_mapper_indep, results))
    df_results = pd.DataFrame(averaged_results)
    return (df_per_trial_results, df_results)

def train_big_models_independent(model_type, hyperparameter_configs, data):
    results = []
    for i, model_config in enumerate(hyperparameter_configs):
        print('Config {}'.format(i))
        current_result = {}
        current_result['model_config'] = model_config
        current_result['left_validation_rmse'] = []
        current_result['right_validation_rmse'] = []
        subjects = data.keys()
        for test_subject in subjects:
            dataset = get_dataset_independent(model_type, data, model_config['window_size'], test_subject)
            model = create_big_model_subject(model_config.copy(), dataset)
            model.summary()
            early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)
            model_hist = model.fit(dataset['X_train'], dataset['y_train'], verbose=1, validation_split=0.2, shuffle=True, callbacks= [early_stopping_callback], **model_config['training'])

            predictions = model.predict(dataset['X_test'])
            left_rmse, right_rmse = custom_rmse(dataset['y_test'], predictions)

            current_result['left_validation_rmse'].append(left_rmse)
            current_result['right_validation_rmse'].append(right_rmse)
            clear_session()
        results.append(current_result)
    
    per_trial_results = []
    for trial in results:
        left_val_rmse = trial['left_validation_rmse']
        right_val_rmse = trial['right_validation_rmse']
        for i, test_subject in enumerate(data.keys()):
            trial_result = {}
            trial_result['test_subject'] = test_subject
            trial_result['window_size'] = trial['model_config']['window_size']

            if (model_type != 'mlp'):
                for key in trial['model_config'][model_type].keys():
                    trial_result['{}_{}'.format(model_type, key)] = trial['model_config'][model_type][key]

            for key in trial['model_config']['dense'].keys():
                trial_result['dense_{}'.format(key)] = trial['model_config']['dense'][key]

            for key in trial['model_config']['optimizer'].keys():
                trial_result['optim_{}'.format(key)] = trial['model_config']['optimizer'][key]

            for key in trial['model_config']['training'].keys():
                trial_result['training_{}'.format(key)] = trial['model_config']['training'][key]
            trial_result['left_validation_rmse'] = left_val_rmse[i-1]
            trial_result['right_validation_rmse'] = right_val_rmse[i-1]
            per_trial_results.append(trial_result)
    df_per_trial_results = pd.DataFrame(per_trial_results)

    for model in results:
        model['left_rmse_mean'] = np.mean(model['left_validation_rmse'])
        model['right_rmse_mean'] = np.mean(model['right_validation_rmse'])
                
    averaged_results = list(map(results_mapper_indep, results))
    df_results = pd.DataFrame(averaged_results)
    return (df_per_trial_results, df_results)

def deploy_big_models_independent(model_type, hyperparameter_configs, data):
    results = []
    for i, model_config in enumerate(hyperparameter_configs):
        dataset = deploy_cnn_extract_features_independent(data, model_config['window_size'])
        model = create_big_model_subject(model_config.copy(), dataset)
        model.summary()
        early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)
        model_history = model.fit(dataset['X_train'], dataset['y_train'], verbose=1, validation_split=0.2, shuffle=True, callbacks= [early_stopping_callback], **model_config['training'])
        training_history = {}
        # training_history['accuracy'] = model_history['acc']
        # training_history['val_accuracy'] = model_history.history['val_acc']
        training_history['loss'] = model_history.history['loss']
        training_history['val_loss'] = model_history.history['val_loss']
    return model, training_history