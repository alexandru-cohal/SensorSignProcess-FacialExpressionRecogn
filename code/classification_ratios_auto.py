import numpy as np
import matplotlib.pyplot as plt
import classification_mlp as class_mlp
import plotRawData
import low_filter
import classification_threshold as class_th
import features_extractor as f_ex
import math
import pyeeg
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

label_dictionary = {"neutral":0, "bl":1, "br":2, "bb":3, "fm":4, "om":5, "eb":6, "m2l":7, "m2r":8, "n":9, "s":10}
path_recordings = 'rawdata_01-03-18/'
PERSON = 'v'
N_SAMPLES_IN_RECORDING = 100

WINDOW_SIZE = 25 
OVERLAPPING_SIZE = 15 

NO_FILES_NEUTRAL_MEAN = 2


def compute_mean_neutral_each_channel(data):
	# Compute the mean of the first NO_FILES_NEUTRAL_MEAN files for each channel of the 'neutral' class 

	mean_channels_neutral = 0
	for index_file in range(NO_FILES_NEUTRAL_MEAN):
		mean_channels_neutral = mean_channels_neutral + np.average(data[label_dictionary["neutral"]][index_file], axis=0)

	mean_channels_neutral = mean_channels_neutral / NO_FILES_NEUTRAL_MEAN

	return mean_channels_neutral


def ratio_windowMean_neutralMean(data_window, mean_neutral):
	# Calculate the ratio between the mean of the given window and the mean of the neutral state

	mean_window = np.average(data_window)
	ratio = mean_window / mean_neutral
	return ratio


def store_ratio_windowMean_neutralMean(data, mean_channels_neutral, classes_used_list, channels_used_list):
	# Calculate the ratios for the specified channels and classes
	# Create the features and labels matrices for training and testing

	no_files_training = 15

	features = []
	labels = []
	
	for file_index in range(20):		
		for class_label in classes_used_list:
			class_index = label_dictionary[class_label]
			for window_start_index in range(0, N_SAMPLES_IN_RECORDING, WINDOW_SIZE - OVERLAPPING_SIZE):
				if window_start_index + WINDOW_SIZE <= N_SAMPLES_IN_RECORDING:
					recording_window = data[class_index][file_index][window_start_index : window_start_index + WINDOW_SIZE]
					
					features_list = []
					for channel_index in channels_used_list:
						channel_ratio_current = ratio_windowMean_neutralMean(recording_window[:,channel_index], mean_channels_neutral[channel_index])
						features_list.append(channel_ratio_current)
						'''
						mobility, complexity = pyeeg.hjorth(recording_window[:,channel_index])
						features_list.append(mobility)
						features_list.append(complexity)
						'''
					'''
					p1, p2 = f_ex.power_ratio(recording_window)
					features_list.append(p1)
					features_list.append(p2)
					'''
					features.append(np.asarray(features_list))
					labels.append(class_index)

	features = np.asarray(features)
	labels = np.asarray(labels)

	no_samples = features.shape[0]
	border_index = int(no_samples * 0.75)

	features_train = features[0 : border_index]
	features_test = features[border_index : no_samples]
	labels_train = labels[0 : border_index]
	labels_test = labels[border_index : no_samples]

	return features_train, labels_train, features_test, labels_test


def train_test_MLP_ratio(features_train, labels_train, features_test, labels_test):
	print "------ MLP ------"

	classifier = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(17,), activation = 'logistic', learning_rate_init = 1, verbose=True)
	classifier.fit(features_train, labels_train)

	pred = classifier.predict(features_test)

	print "Confusion matrix - testing:"
	print confusion_matrix(labels_test, pred)
	print "Accuracy - training:"
	print classifier.score(features_train, labels_train)
	print "Accuracy - testing:"
	print classifier.score(features_test, labels_test)
		
	
def train_test_SVM_ratio(features_train, labels_train, features_test, labels_test):
	print "------ SVM ------"

	classifier = svm.SVC()
	classifier.fit(features_train, labels_train)

	pred = classifier.predict(features_test)

	print "Confusion matrix - testing:"
	print confusion_matrix(labels_test, pred)
	print "Accuracy - training:"
	print classifier.score(features_train, labels_train)
	print "Accuracy - testing:"
	print classifier.score(features_test, labels_test)


def train_test_KNN_ratio(features_train, labels_train, features_test, labels_test):
	print "------ KNN ------"

	classifier = KNeighborsClassifier(n_neighbors = 31)
	classifier.fit(features_train, labels_train)

	pred = classifier.predict(features_test)
	'''
	print "Confusion matrix - testing:"
	print confusion_matrix(labels_test, pred)
	print "Accuracy - training:"
	print classifier.score(features_train, labels_train)
	print "Accuracy - testing:"
	print classifier.score(features_test, labels_test)
	'''
	return classifier.score(features_train, labels_train), classifier.score(features_test, labels_test)

def train_test_GNB_ratio(features_train, labels_train, features_test, labels_test):
	print "------ GNB ------"

	classifier = GaussianNB()
	classifier.fit(features_train, labels_train)

	pred = classifier.predict(features_test)
	
	print "Confusion matrix - testing:"
	print confusion_matrix(labels_test, pred)
	print "Accuracy - training:"
	print classifier.score(features_train, labels_train)
	print "Accuracy - testing:"
	print classifier.score(features_test, labels_test)
	'''
	return classifier.score(features_train, labels_train), classifier.score(features_test, labels_test)
	'''

def main():
	# Read the input data (recordings)
	data = class_mlp.read_recordings(path_recordings, PERSON)

	# Compute the average of neutral for each channel
	mean_channels_neutral = compute_mean_neutral_each_channel(data)

	# Normalize the input data using the neutral mean value
	#data = class_th.normalize_data(data, mean_channels_neutral)

	# Compute the features and create features and labels arrays
	#classes_used_list = ("neutral", "bl", "br", "bb", "om", "fm")
	classes_used_list_list = (("neutral", "bb"), ("neutral", "bb", "br", "bl"), ("neutral", "bb", "br", "bl", "om", "fm"))
	channels_used_list_list = ((2,3), (2,3,10), (2,3,10,11), (2,3,10,11,12,13), range(14))

	acc_matrix = []
	for classes_used_list in classes_used_list_list:
		print 'Classes: ', classes_used_list 
		acc_row = []

		for channels_used_list in channels_used_list_list:
			print '		Channels: ', channels_used_list
			features_train, labels_train, features_test, labels_test = store_ratio_windowMean_neutralMean(data, mean_channels_neutral, classes_used_list, channels_used_list)

			# Train and test different classifiers
			#train_test_MLP_ratio(features_train, labels_train, features_test, labels_test)
			#train_test_SVM_ratio(features_train, labels_train, features_test, labels_test)
			#acc_train, acc_test = train_test_KNN_ratio(features_train, labels_train, features_test, labels_test)
			train_test_GNB_ratio(features_train, labels_train, features_test, labels_test)

			#acc_row.append(acc_train)
			#acc_row.append(acc_test)

		acc_matrix.append(acc_row)
			
	np.save('knn_k_31_accuracy.npy', acc_matrix);
	print acc_matrix
	

if __name__ == "__main__":
    main()

