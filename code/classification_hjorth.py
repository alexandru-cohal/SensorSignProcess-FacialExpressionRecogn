import numpy as np
import matplotlib.pyplot as plt
import classification_mlp as class_mlp
import plotRawData
import low_filter
import math
import pyeeg
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

label_dictionary = {"neutral":0, "bl":1, "br":2, "bb":3, "fm":4, "om":5, "eb":6, "m2l":7, "m2r":8, "n":9, "s":10}
path_recordings = 'rawdata_01-03-18/'
PERSON = 'v'
N_SAMPLES_IN_RECORDING = 100

WINDOW_SIZE = 25 
OVERLAPPING_SIZE = 15 


def store_hjorth(data, classes_used_list, channels_used_list):
	# Calculate the Hjorth parameters for the specified channels and classes
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
					
					channels_hjorth_list = []
					for channel_index in channels_used_list:
						mobility, complexity = pyeeg.hjorth(recording_window[:,channel_index])
						channels_hjorth_list.append(mobility)
						channels_hjorth_list.append(complexity)
					
					features.append(np.asarray(channels_hjorth_list))
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

	classifier = MLPClassifier(solver="lbfgs", hidden_layer_sizes=(5,), activation = 'logistic', learning_rate_init = 1, verbose=True)
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

	classifier = KNeighborsClassifier(n_neighbors = 5)
	classifier.fit(features_train, labels_train)

	pred = classifier.predict(features_test)

	print "Confusion matrix - testing:"
	print confusion_matrix(labels_test, pred)
	print "Accuracy - training:"
	print classifier.score(features_train, labels_train)
	print "Accuracy - testing:"
	print classifier.score(features_test, labels_test)


def main():
	# Read the input data (recordings)
	data = class_mlp.read_recordings(path_recordings, PERSON)

	# Compute the features and create features and labels arrays
	classes_used_list = ("neutral", "bl", "br", "bb", "om", "fm")
	channels_used_list = range(14)

	features_train, labels_train, features_test, labels_test = store_hjorth(data, classes_used_list, channels_used_list)

	# Train and test different classifiers
	#train_test_MLP_ratio(features_train, labels_train, features_test, labels_test)
	#train_test_SVM_ratio(features_train, labels_train, features_test, labels_test)
	train_test_KNN_ratio(features_train, labels_train, features_test, labels_test)


if __name__ == "__main__":
    main()

