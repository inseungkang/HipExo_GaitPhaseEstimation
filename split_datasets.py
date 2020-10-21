from utils import *
import pickle
import pandas as pd

path = 'data/strokeData/'
headers = pd.read_csv('data/strokeData/field_v3.txt')
subject = 'ST05'
assistance_levels = ['L0R0', 'L0R1', 'L1R1', 'L2R1', 'L2R2', 'L3R1']
window_size = 80

# Split All Datasets
for level in assistance_levels:
	filename = path+subject+'/windowed_' + level
	with open(filename+'.pkl', 'rb') as f:
		data = pickle.load(f)
		split_windowed_dataset(data, 'leftGaitPhase', filename)
		f.close()