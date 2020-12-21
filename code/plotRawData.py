# Plot all the channels of the raw data in time domain

import numpy as np
import matplotlib.pyplot as plt
import classification_mlp
import matplotlib.patches as mpatches

def plot_all_channels(data_array, title = 'Plot of all channels', no_neutral_clf = []):
	n_samples = data_array.shape[0]

        color_list = ["white", "red", "blue", "magenta", "orange", "yellow"] 
        handles = [mpatches.Patch(color = "red", label = "Bink Left"), \
                   mpatches.Patch(color = "blue", label = "Bink Right"), \
                   mpatches.Patch(color = "magenta", label = "Blink Both"), \
                   mpatches.Patch(color = "orange", label = "Full Mouth"), \
                   mpatches.Patch(color = "yellow", label = "Open Mouth")]

	fig = plt.figure()
	fig.suptitle(title)
	for i in range(14):   
		ax = plt.subplot(7, 2, i+1)
		ax.plot(range(0,n_samples), data_array[:,i])
                for sample_index, sample_value  in no_neutral_clf:
                    plt.axvspan(sample_index - 1 , sample_index + 1 , color= color_list[sample_value], alpha=0.5)
		ax.set_title(str(i))
		ax.grid()	

        plt.figlegend(handles = handles, loc = 1)
	plt.show()   

def main():
	basePath = 'rawdata_01-03-18/'
        person = 'v'
        whole_data = classification_mlp.read_recordings(basePath, person)
        class_label = 'bl'
        class_index = classification_mlp.label_dictionary[class_label]
        file_index = 12
        data_array = whole_data[class_index][file_index] 
        title = "class " + class_label + " file " + str(file_index)

	plot_all_channels(data_array, title)

if __name__ == "__main__":
    main()
