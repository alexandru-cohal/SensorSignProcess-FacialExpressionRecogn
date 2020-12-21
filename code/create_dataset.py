import numpy as np

#######################################################################################
def concat_data_files(file_path, file_name_begin, no_recordings, label_value):

	data = np.empty((0, 14))
	for index_file in range(no_recordings):
		file_name_current = file_name_begin + str(index_file) + '.npy'
		data_current = np.load(file_path + file_name_current)

		data = np.vstack((data, data_current))
		
	label = label_value * np.ones(data.shape[0])

return data, label

#######################################################################################
def split_train_test(data, label, percentage):
	no_train = int(percentage * data.shape[0] / 100)
	

#######################################################################################
file_path = 'rawdata_01-03-18/'
file_name_begin = 'a_bb_'
no_recordings = 20
label_value = 2

