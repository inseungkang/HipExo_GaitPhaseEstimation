import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pickle

def window_data(data, window_size, ground_truth_cols):
	"""Takes dataset and creates backward looking windows associated with each timestep
	Args:
		data (DataFrame): dataset of shape (M datapoints, K features + P ground truth)
		window_size (int): window size
		ground_truth_cols ([string]): ground truth column names
	Returns:
		windowed_data: (M, window_size, K + P)
	"""

	num_cols = len(data.columns)
	P = len(ground_truth_cols)
	K = num_cols - P

	feature_cols = []
	for col in data.columns:
		if col not in ground_truth_cols:
			feature_cols.append(col)

	features = data[feature_cols].to_numpy()
	ground_truth = data[ground_truth_cols].to_numpy()

	#Sliding window
	shape_des = (features.shape[0] - window_size +
					1, window_size, features.shape[-1])
	strides_des = (
		features.strides[0], features.strides[0], features.strides[1])
	X = np.lib.stride_tricks.as_strided(features, shape=shape_des,
												strides=strides_des)
	Y = ground_truth[window_size-1:]

	windowed_dataset = {
		'X': X,
		'Y': Y,
		'feature_cols': feature_cols,
		'ground_truth_cols': ground_truth_cols,
		'window_size': window_size
	}

	return windowed_dataset

def convert_to_gp(x, y):
	"""Take cartesian gait phase values (x, y) and convert to percentage"""
	gp = np.mod(np.arctan2(y, x)+2*np.pi, 2*np.pi) / (2*np.pi)
	return gp

def split_dataset_pandas(data, column, filename):
	"""Take a pandas dataframe, plot a column, and split the dataset into two parts
	"""

	def onclick(event):
		print(event.xdata)

	print('Click to view location')

	# Plot Column
	f = plt.figure(figsize=(10, 4))
	plt.plot(data[column])
	f.canvas.mpl_connect('button_press_event', onclick)
	plt.show()    

	split_ind = input("Enter integer index of dataset split\n")

	left_clip = data.loc[:split_ind]
	right_clip = data.loc[split_ind:]

	# Save all data for this trial
	left_clip.to_csv(filename+f'_pt1')
	right_clip.to_csv(filename+f'_pt2')
	print('Split and Saved')

def split_windowed_dataset(dataset, column, filename):
	"""Take a windowed dataset dictionary, plots a ground truth column,
	and split the dataset into two parts
	"""

	def onclick(event):
		print(event.xdata)

	print('Click to view location')

	if column in dataset['ground_truth_cols']:
		data = dataset['Y']
		col_ndx = dataset['ground_truth_cols'].index(column)
	else:
		print('Column not in dataset')

	# Plot Column
	f = plt.figure(figsize=(10, 4))
	plt.plot(data[:,col_ndx])
	f.canvas.mpl_connect('button_press_event', onclick)
	plt.show()

	split_ind = (int)(input("Enter integer index of dataset split\n"))

	left_clip = {
		'X': dataset['X'][:split_ind],
		'Y': dataset['Y'][:split_ind],
		'feature_cols': dataset['feature_cols'],
		'ground_truth_cols': dataset['ground_truth_cols'],
		'window_size': dataset['window_size']
	}

	right_clip = {
		'X': dataset['X'][split_ind:],
		'Y': dataset['Y'][split_ind:],
		'feature_cols': dataset['feature_cols'],
		'ground_truth_cols': dataset['ground_truth_cols'],
		'window_size': dataset['window_size']
	}

	# Save all data for this trial
	with open(filename+f'_pt1.pkl', 'wb') as f:
		pickle.dump(left_clip, f)
		f.close()

	with open(filename+f'_pt2.pkl', 'wb') as f:
		pickle.dump(right_clip, f)
		f.close()
	print('Split and Saved')

def create_training_set(filenames):
	# Aggregate All Training Datasets
	X_train = None
	Y_train = None
	window_size = None
	feature_cols = None
	ground_truth_cols = None

	for filename in filenames:
		with open(filename+'.pkl', 'rb') as f:
			data = pickle.load(f)
			if (type(X_train) == type(None)):
				X_train = data['X']
				Y_train = data['Y']
				window_size = data['window_size']
				feature_cols = data['feature_cols']
				ground_truth_cols = data['ground_truth_cols']
			else:
				X_train = np.concatenate((X_train, data['X']), axis=0)
				Y_train = np.concatenate((Y_train, data['Y']), axis=0)
			f.close()

	training_dataset = {
		'X': X_train,
		'Y': Y_train,
		'feature_cols': feature_cols,
		'ground_truth_cols': ground_truth_cols,
		'window_size': window_size
	}

	return training_dataset