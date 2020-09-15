from data_processing import *
from convolutional_nn import *
#TODO: change convolutional_nn to the training file that the final pipeline uses

data = import_data()
data = label_data(data)
data = cut_data(data)
###################### Training Neural Network ##################
window_sizes = np.arange(20, 221, 20)

train_cnn(data_list, window_sizes, [Adam()])
plot_err(window_sizes * 5)
