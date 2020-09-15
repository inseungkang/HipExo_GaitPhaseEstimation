import itertools

def get_model_configs(hyperparam_space):
    model_configs = []
    model_type = hyperparam_space['model']
    for window_size in hyperparam_space['window_size']:
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

        possible_configs = itertools.product(type_specific_configs, dense_configs, optim_configs)
        config_count = 0
        for config in possible_configs:
            config_count += 1
            config_obj = {
                'window_size': window_size,
                'dense': config[1],
                'optimizer': config[2]
            }
            config_obj[model_type] = config[0]
            model_configs.append(config_obj)
    return model_configs

def train_models(model_type, hyperparameter_configs):
    for model_config in hyperparameter_configs:
        current_result = {}
        current_result['model_config'] = model_config
        current_result['left_validation_rmse'] = []
        current_result['right_validation_rmse'] = []
        for trial in np.arange(1,11):
            dataset = cnn_extract_features(data_list, model_config['window_size'], trial)
            model = lstm_model(sequence_length=model_config['window_size'],
                              n_features=10, 
                               lstm_config=model_config['lstm'],
                               dense_config=model_config['dense'],
                               optim_config=model_config['optimizer'],
                               X_train=dataset['X_train'].squeeze())
            model.summary()
            early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=0)
            model_hist = model.fit(dataset['X_train'].squeeze(), dataset['y_train'].squeeze(), epochs=30, batch_size=128, verbose=1, validation_split=0.2, shuffle=True, callbacks= [early_stopping_callback])
            # model.save('test_rnn_model')

            predictions = model.predict(dataset['X_test'].squeeze())
            left_rmse, right_rmse = custom_rmse(dataset['y_test'].squeeze(), predictions)

            current_result['left_validation_rmse'].append(left_rmse)
            current_result['right_validation_rmse'].append(right_rmse)
            clear_session()
        results.append(current_result)
        