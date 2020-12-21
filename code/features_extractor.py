import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import pyeeg
import classification_mlp as clf


def power_ratio(data_array, bands  = [0,2,5,10], FS = 128.0):
	my_scaling = "spectrum"
	nperseg = data_array.shape[0] 

	f, den = signal.welch(data_array, FS, nperseg = nperseg, axis = 0, scaling = my_scaling)

	#print f

	den_all_channels = den.sum(axis=1)
	n = den_all_channels[bands[0] : bands[1]].sum()
	d = den_all_channels[bands[2] : bands[3]].sum()

	print "F = " , f[bands[0]], f[bands[1]], f[bands[2]], f[bands[3]]

	return n, d 
    

def main():
    path = '/home/vitorroriz/sensorsig/rawdata_01-03-18/'
    FS = 128
    person = 'v'
    class_label = "br"
    class_label2 = "bl"

    recordings = clf.read_recordings(path, person)
    class_index = clf.label_dictionary[class_label]
    class_index2 = clf.label_dictionary[class_label2]
   
    power_ratio_list = []
    power_ratio_list_2 = []

    hjorth_m_list = []
    hjorth_m_list2 = []
    hjorth_c_list = []
    hjorth_c_list2 = []

    channel = 10 

    for file_index in range(20):
        power_ratio_list.append(power_ratio(recordings[class_index][file_index]))
        power_ratio_list_2.append(power_ratio(recordings[class_index2][file_index]))
        m,c = pyeeg.hjorth(recordings[class_index][file_index][:,channel])
        hjorth_m_list.append(m)
        hjorth_c_list.append(c)
        m2,c2 = pyeeg.hjorth(recordings[class_index2][file_index][:,channel])
        hjorth_m_list2.append(m2)
        hjorth_c_list2.append(c2)

    plt.plot(hjorth_m_list, 'bo')
    plt.hold(True)
    plt.plot(hjorth_m_list2, 'rx')
    plt.show()

if __name__ == "__main__":
    main()
