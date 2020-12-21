import os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import low_filter as lfilt

label_dictionary = {"neutral":0, "bl":1, "br":2, "bb":3, "fm":4, "om":5, "eb":6, "m2l":7, "m2r":8, "n":9, "s":10}
N_CHANNELS = 14
N_SAMPLES_IN_RECORDING = 100 
DC_VALUE = 1250

def read_recordings(path, person):
	# Read all the recording files (each file contains one matrix 110 by 14) for a specified person and store them  
	# in a list of lists according to the action type (class)

	data = [[] for x in range( len(label_dictionary) )]

	for file_name in sorted(os.listdir(path)):
		# Select only the recordings of the specified person
		if file_name[0] == person:
			# Get the class label (as string) of the current recording from the file name
			file_label = file_name.split("_")[1]
			# Get the index of the class label (as int) using the class label dictionary
			index_label = label_dictionary[file_label]

			# Load the current recording
			data_current = np.load(path + file_name)
			
			# Add the recording data to the corresponding list
			data_current = data_current - DC_VALUE
                        cut_frequency = 16 
			data_current = lfilt.low_filter_channel_all(data_current, cut_frequency = cut_frequency)
			data[index_label].append(data_current)

	return data
	
	
def create_train_test_dataset(data, class_label, train_test_ratio, window_size, overlapping_size):
	# Create the training and the testing datasets from the recordings stored in 'data'
	# 'train_test_ratio' (real value between 0 and 1): defines the number of recording used for training and for testing
	# 'window_size': defines the number of recorded samples used together (in a window) as inputs for the Neural Network
	# 'overlapping_size': defines the number of recorded samples with which two consecutive windows overlap

	class_index = label_dictionary[class_label]

	# Number of recordings (files) used for training and for testing
	n_rec_train = int( len(data[class_index]) * train_test_ratio )
	n_rec_test = len(data[class_index]) - n_rec_train

	# Lists of recordings (matrices) for training and for testing
	data_class_train = data[class_index][0:n_rec_train]
	data_class_test  = data[class_index][n_rec_train:n_rec_train + n_rec_test]

	# Bring the training dataset in matrix format (lines: samples, columns: features)
	# Each sample (line) has window_size consecutive values for each channel
	data_train_list = []
	for recording in data_class_train:
		for window_start_index in range(0, N_SAMPLES_IN_RECORDING, window_size - overlapping_size):
			if window_start_index + window_size <= N_SAMPLES_IN_RECORDING:
				recording_window = recording[window_start_index : window_start_index + window_size]
				sample = recording_window.flatten('F')
				data_train_list.append(sample)
	data_train_matrix = np.asarray(data_train_list)
	label_train = class_index * np.ones(data_train_matrix.shape[0])

	# Bring the testing dataset in matrix format (lines: samples, columns: features)
	# Each sample (line) has window_size consecutive values for each channel
	data_test_list = []
	for recording in data_class_test:
		for window_start_index in range(0, N_SAMPLES_IN_RECORDING, window_size - overlapping_size):
			if window_start_index + window_size <= N_SAMPLES_IN_RECORDING:
				recording_window = recording[window_start_index : window_start_index + window_size]
				sample = recording_window.flatten('F')
				data_test_list.append(sample)
	data_test_matrix = np.asarray(data_test_list)
	label_test = class_index * np.ones(data_test_matrix.shape[0])

	return data_train_matrix, label_train, data_test_matrix, label_test


def concatenate_train_test_datasets_multiple_classes(labels_to_train, data):
	for label in labels_to_train:
		data_train, label_train, data_test, label_test  = create_train_test_dataset(data, label, 0.2, 50, 0)
		
		try:
			data_train_all
		except:
			data_train_all = data_train
			label_train_all = label_train
			data_test_all = data_test
			label_test_all = label_test
		else:
			data_train_all = np.vstack((data_train_all, data_train))
			label_train_all = np.hstack((label_train_all, label_train))
			data_test_all = np.vstack((data_test_all, data_test))
			label_test_all = np.hstack((label_test_all, label_test))

	return data_train_all, label_train_all, data_test_all, label_test_all


def training_classifier(data_train, label_train, data_test, label_test):
	clf = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(70,), activation = 'logistic', learning_rate_init = 1, verbose=True)

	clf.fit(data_train, label_train)

	pred = clf.predict(data_test)

	print "Confusion matrix:"
	print confusion_matrix(label_test, pred)
	print "Accuracy:"
	print clf.score(data_test, label_test)


def main():		
	path_recordings = 'rawdata_01-03-18/'
	person = 'v'

	data = read_recordings(path_recordings, person)	
	
	labels_to_train = ("neutral", "br", "bl")

	data_train, label_train, data_test, label_test = concatenate_train_test_datasets_multiple_classes(labels_to_train, data)

	training_classifier(data_train, label_train, data_test, label_test)
	

if __name__ == "__main__":
    main()
