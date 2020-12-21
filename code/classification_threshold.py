import numpy as np
import matplotlib.pyplot as plt
import classification_mlp as class_mlp
import plotRawData
import low_filter
import math

label_dictionary = {"neutral":0, "bl":1, "br":2, "bb":3, "fm":4, "om":5, "eb":6, "m2l":7, "m2r":8, "n":9, "s":10}
path_recordings = 'rawdata_01-03-18/'
PERSON = 'v'
N_SAMPLES_IN_RECORDING = 100

WINDOW_SIZE = 25 
OVERLAPPING_SIZE = 0 
N_SAMPLES_TO_IGNORE = 3 

NO_FILES_NEUTRAL_MEAN = 2


def compute_mean_neutral_each_channel(data):
	# Compute the mean of the first NO_FILES_NEUTRAL_MEAN files for each channel of the 'neutral' class 

	mean_channels_neutral = 0
	for index_file in range(NO_FILES_NEUTRAL_MEAN):
		mean_channels_neutral = mean_channels_neutral + np.average(data[label_dictionary["neutral"]][index_file], axis=0)

	mean_channels_neutral = mean_channels_neutral / NO_FILES_NEUTRAL_MEAN

	return mean_channels_neutral


def normalize_data(data, mean_channels_neutral):
	# Normalize all the files of each class by the mean of the 'neutral' class

	for class_index in range(11):
		for file_index in range(20): 
			data[class_index][file_index] = data[class_index][file_index] / mean_channels_neutral

	return data


def ratio_windowMean_neutralMean(data_window, mean_neutral):
	# Calculate the ratio between the mean of the given window and the mean of the neutral state

	mean_window = np.average(data_window)
	ratio = mean_window / mean_neutral
	return ratio


def channel_drop(data_window, mean_neutral, threshold):
	# Detect a drop in only 1 channel

	if ratio_windowMean_neutralMean(data_window, mean_neutral) < threshold:
		return True
	return False

def channel_peak(data_window, mean_neutral, threshold):
	# Detect a peak in only 1 channel

	if ratio_windowMean_neutralMean(data_window, mean_neutral) > threshold:
		return True
	return False

def window_threshold_classifier(data_window, mean_channels_neutral):
    ch2_ratio = ratio_windowMean_neutralMean(data_window[:,2], mean_channels_neutral[2]) 
    ch3_ratio = ratio_windowMean_neutralMean(data_window[:,3], mean_channels_neutral[3]) 
    ch10_ratio = ratio_windowMean_neutralMean(data_window[:,10], mean_channels_neutral[10]) 

    print "ratios = ", ch2_ratio, ch3_ratio, ch10_ratio
	
    if(ch3_ratio < 0.95):
        print "CH3 smaller than 0.95"
        if(ch3_ratio < 0.75):
            print "CH3 smaller than 0.75"
            print "returning om"
            return label_dictionary["om"] 
        else:
            print "CH3 bigger than 0.75"
            print "returning fm"
            return label_dictionary["fm"]
    else:
        print "CH3 bigger than 0.95"
        if(ch10_ratio > 1.16):
            print "CH10 bigger than 1.16"
            if(ch2_ratio > 1.01):
                print "CH2 bigger than 1.01"
                if(ch2_ratio > 1.12):
                    print " CH2 bigger than 1.15"
                    if(ch3_ratio > 1.15):
                        print "CH3 bigger than 1.15"
                        print "returning bb"
                        return label_dictionary["bb"]
                    else:
                        "CH3 smaller than 1.15"
                        print "returning neutral"
                        return label_dictionary["neutral"]
                else:
                    print "CH2 smaller than 1.15"
                    print "returning bl"
                    return label_dictionary["bl"]
            else:
                print "CH2 smaller than 1.01"
                print "returning br"
                return label_dictionary["br"]
        else:
            print "CH10 smaller than 1.16"
            print "returning neutral"
            return label_dictionary["neutral"]

def classification_filter(classification, samples_counter):
    if(samples_counter == -1):
        if(classification != label_dictionary["neutral"]):
                samples_counter = samples_counter+1
                return classification, samples_counter 
        else:
               return label_dictionary["neutral"], samples_counter 
    else:
        if(samples_counter < N_SAMPLES_TO_IGNORE - 1):
                samples_counter = samples_counter+1
                return label_dictionary["neutral"], samples_counter 
        else:
                samples_counter = -1
                return label_dictionary["neutral"], samples_counter 

def plot_ratio_all_channels_all_recordings(data, mean_channels_neutral, class_label, min_channel, max_channel):
	# Plot the normalized windows for all recordings, channel by channel

	class_index = label_dictionary[class_label]
	no_channels = max_channel - min_channel + 1

	fig = plt.figure()
	fig.suptitle('class: ' + class_label + ' (xAxis: window index, yAxis: normalized window mean)')
	 
	for channel in range(min_channel, max_channel+1):
		for file_index in range(20):
			data_1channel = data[class_index][file_index][:,channel]

			ratio_list = []
			drop_status_list = []
			for window_start_index in range(0, N_SAMPLES_IN_RECORDING, WINDOW_SIZE - OVERLAPPING_SIZE):
				if window_start_index + WINDOW_SIZE <= N_SAMPLES_IN_RECORDING:
					recording_window = data_1channel[window_start_index : window_start_index + WINDOW_SIZE]
					ratio = ratio_windowMean_neutralMean(recording_window, mean_channels_neutral[channel])
					ratio_list.append(ratio)

			if no_channels == 1:
				no_columns_subplot = 1
			else:
				no_columns_subplot = 2
			no_rows_subplot	= math.ceil(no_channels / 2.0)
		
			ax = plt.subplot(no_rows_subplot, no_columns_subplot, channel-min_channel+1)
			ax.plot(ratio_list)
			ax.set_title(str(channel))
			ax.grid(True)

	# fig_name = 'ratio_windowMean_neutralMean_person_' + PERSON + '_class_' + class_label + '_windowSize_' + str(WINDOW_SIZE) + '_overlappingSize_' + str(OVERLAPPING_SIZE) + '.png'
	# plt.savefig('plots/' + fig_name)

	plt.show()

def main():
	# Read the input data (recordings)
	data = class_mlp.read_recordings(path_recordings, PERSON)

	# Compute the average of neutral for each channel
	mean_channels_neutral = compute_mean_neutral_each_channel(data)

	# Normalize the input data using the neutral mean value
	# data = normalize_data(data, mean_channels_neutral)

	# Plot the normalized windows for all recordings, channel by channel
        class_label = "bl"
        channel_ut = 2
#	plot_ratio_all_channels_all_recordings(data, mean_channels_neutral, class_label, channel_ut, channel_ut)

        file_index = 7 
        class_index = label_dictionary[class_label]
        recording = data[class_index][file_index]
	
        classification_list  = []
        raw_classification_list  = []
        clf_sample_indexes = []

        samples_counter = -1
        for window_start_index in range(0, N_SAMPLES_IN_RECORDING, WINDOW_SIZE - OVERLAPPING_SIZE):
                if window_start_index + WINDOW_SIZE <= N_SAMPLES_IN_RECORDING:
                        recording_window = recording[window_start_index : window_start_index + WINDOW_SIZE]
                        print ""
                        print "########################################################"
                        print "Classifying window ", window_start_index
                        raw_classification = window_threshold_classifier(recording_window, mean_channels_neutral)
                        classification, samples_counter = classification_filter(raw_classification, samples_counter)
                        classification_list.append(classification)
                        raw_classification_list.append(raw_classification)

                        clf_sample_indexes.append(window_start_index)
        
        no_neutral_clf = [(clf_sample_indexes[sample_index], sample_value)  for sample_index, sample_value in enumerate(classification_list) if sample_value != 0] 
       
#        no_neutral_samples = [(clf_sample_indexes[index], value) for index, value in no_neutral_clf]
        title = "Class " + class_label + " File " + str(file_index)
        plotRawData.plot_all_channels(recording, title, no_neutral_clf)         

if __name__ == "__main__":
    main()

