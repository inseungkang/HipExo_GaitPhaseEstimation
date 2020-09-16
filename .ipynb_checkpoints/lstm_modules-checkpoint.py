from tensorflow.keras.backend import clear_session
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from tensorflow.keras.layers.experimental.preprocessing import Normalization 

def lstm_model(sequence_length, n_features, X_train):
    model = Sequential()
    norm_layer = Normalization(input_shape=(sequence_length, n_features))
    norm_layer.adapt(X_train)
    model.add(norm_layer)
    model.add(LSTM(30, return_sequences = False, activation='relu'))
    model.add(Dense(4, activation='tanh'))
    model.compile(loss='mean_squared_error', optimizer='rmsprop')
    return model

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