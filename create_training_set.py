from utils import *
import pickle
import pandas as pd

path = 'data/strokeData/'
headers = pd.read_csv('data/strokeData/field_v3.txt')
subject = 'ST05'
assistance_levels = ['L0R0', 'L0R1', 'L1R1', 'L2R1', 'L2R2', 'L3R1']

# Aggregate All Training Datasets
X_train = None
Y_train = None
window_size = None
feature_cols = None
ground_truth_cols = None

filenames = [path+subject+'/windowed_' + level + '_pt1' for level in assistance_levels]
training_dataset = create_training_set(filenames)

# Save all data for this trial
with open(path+subject+'/training.pkl', 'wb') as f:
	pickle.dump(training_dataset, f)
	f.close()