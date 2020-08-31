from keras.models import load_model
from convolutional_nn import *

trials = np.arange(1, 11)

model = load_model('test_model_save')
window_size = 20
loss_per_trial = []
for test_trial_num in trials:
    data = cnn_train_test_split(test_trial_num, window_size)
    loss_per_trial.append(model.evaluate(
            data['X_test'], data['y_test']))
#     y_preds = model.predict(data['X_test'])
    # writes the predictions to a file in predictions file
    # file_name = f'predictions/{mode}_wsize{window_size}_{num_layer}layers_{num_node}nodes_optimizer{ix+1}_trial{test_trial_num}.txt'
    # y_preds.to_csv(file_name, index=False)
loss_mean = np.mean(loss_per_trial) * 100
print('Window Size: {} \nAccuracy: {:.2f}%'.format(
    window_size, loss_mean))