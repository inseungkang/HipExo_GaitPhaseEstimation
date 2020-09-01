from keras.models import load_model
from convolutional_nn import *

model = load_model('test_model_save')

# Check that loss is the same as trained model

trials = np.arange(1, 11)
window_size = 20
loss_per_trial = []
for test_trial_num in trials:
    data = cnn_train_test_split(test_trial_num, window_size)
    loss_per_trial.append(model.evaluate(
            data['X_test'], data['y_test']))
loss_mean = np.mean(loss_per_trial) * 100
print('Window Size: {} \Loss: {:.2f}%'.format(
    window_size, loss_mean))