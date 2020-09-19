from data_processing import *
from model_training import *

data = segment_data()
# data = label_data(data)
# data = cut_data(data)
# ###################### Training Neural Network ##################
# window_sizes = np.arange(20, 221, 20)

# train_cnn(data, window_sizes, ['adam'])
# plot_err(window_sizes * 5)
