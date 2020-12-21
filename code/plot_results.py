import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
from matplotlib import colors

def plot_knn_results():
    path = "knn_results_all/"
    file_list = os.listdir(path)
    fig, ax = plt.subplots()
    plt.hold(True)
    ind = range(5)

    jet = plt.cm.jet
    color_list = jet(np.linspace(0, 1, 10))
    
    file_index = 0 
    for file_name in file_list:
        acc_list = []
        current_file = np.load(path+file_name)
        for acc_index in range(1,10,2):
            acc_list.append(current_file[2][acc_index] * 100)
        plt.plot(ind, acc_list, '-D', color = color_list[file_index])     
        file_index=file_index+1

    plt.xlabel("Sets of channels used for feature extraction", fontsize = 18)
    plt.ylabel("Accuracy (%)", fontsize = 18)

    labelsize_ticks = 15
    set_labels = ['(2,3)','(2,3,10)', '(2,3,10,11)', '(2,3,10,11,12,13)', 'All']
    ax.set_xticks(ind)
    ax.set_xticklabels(set_labels, ha="center") 
    plt.tick_params(axis='both', which='major', labelsize=labelsize_ticks)

    title = "All Classes KNN classifier for different values of K"
    plt.suptitle(title, fontsize=20)

    print color_list
    print color_list[0]
#    color_list("r", "b", "k", "m", "y", "g", "o", "c",  
    handles = [mpatches.Patch(color = color_list[0], label = "K = 3"), \
            mpatches.Patch(color = color_list[1], label = "K = 5"), \
            mpatches.Patch(color = color_list[2], label = "K = 7"), \
            mpatches.Patch(color = color_list[3], label = "K = 9"), \
            mpatches.Patch(color = color_list[4], label = "K = 11"), \
            mpatches.Patch(color = color_list[5], label = "K = 13"), \
            mpatches.Patch(color = color_list[6], label = "K = 15"), \
            mpatches.Patch(color = color_list[7], label = "K = 17"), \
            mpatches.Patch(color = color_list[8], label = "K = 19"), \
            mpatches.Patch(color = color_list[9], label = "K = 31")]

    plt.figlegend(handles = handles, loc = 1, fontsize = 'x-large') 
    plt.grid(True)
    plt.show()

def plot_results():
    acc_data = np.load("knn_accuracy.npy")

#   MLP 2 classes 
    acc1 = (91.25, 86.25, 88.33, 85.00, 90.00, 78.75, 93.74, 71.25, 100.00, 100.00) 
#   MLP 4 classes
    acc2 = (61.66, 70.0, 56.45, 60.62, 65.2, 56.25, 68.75, 60.62, 88.54, 80.62)
#   MLP All classes
    acc3 = (54.86, 55.83, 51.38, 57.92, 55.00, 55.83, 60.51, 54.58, 68.75, 79.58)
#   SVM 2 classes
    acc4 = (68.33, 70.00, 68.33, 70.00, 67.50, 70.00, 65.41, 70.00, 60.41, 66.25)
#   SVM 4 classes
    acc5 = (51.88, 59.38, 41.66, 42.50, 41.86, 42.50, 40.20, 41.87, 37.30, 40.00)
#   SVM all classes
    acc6 = (44.30, 51.66, 37.50, 40.41, 37.50, 40.41, 37.08, 40.00, 39.16, 38.75)
#   KNN = 3, 2 classes
    acc7 = acc_data[0,:]
#   KNN = 3, 4 classes
    acc8 = acc_data[1,:]
#   KNN = 3, all classes
    acc9 = acc_data[2,:]
    
    fig, ax = plt.subplots()
    
    ind = (1,2,    7,8,   13,14,   19,20,    25,26)
    set0_train, set0_test, set1_train, set1_test, set2_train, set2_test, set3_train, set3_test, set4_train, set4_test  = plt.bar(ind, acc9) 

    color_test = 'g'
    color_train = 'b'

    set0_train.set_facecolor(color_train)
    set1_train.set_facecolor(color_train)
    set2_train.set_facecolor(color_train)
    set3_train.set_facecolor(color_train)
    set4_train.set_facecolor(color_train)

    set0_test.set_facecolor(color_test)
    set1_test.set_facecolor(color_test)
    set2_test.set_facecolor(color_test)
    set3_test.set_facecolor(color_test)
    set4_test.set_facecolor(color_test)
    
    title1 ="Two Classes MLP Classifier (Neutral, Both Blink)" 
    title2 ="Four Classes MLP Classifier (Neutral, Both Blink, Left Blink, Right Blink)" 
    title3 ="All Classes MLP Classifier (Neutral, Both Blink, Left Blink, Right Blink, Open Mouth, Full Mouth)" 
    title4 ="Two Classes SVM Classifier (Neutral, Both Blink)"
    title5 ="Four Classes SVM Classifier (Neutral, Both Blink, Left Blink, Right Blink)" 
    title6 ="All Classes SVM Classifier (Neutral, Both Blink, Left Blink, Right Blink, Open Mouth, Full Mouth)" 
    title7 ="Two Classes KNN (K = 3) Classifier (Neutral, Both Blink)" 
    title8 ="Four Classes KNN (K = 3) Classifier (Neutral, Both Blink, Left Blink, Right Blink)" 
    title9 ="All Classes KNN (K = 3)"

    plt.suptitle(title9, fontsize=20)
    set_labels = ['(2,3)', '', '(2,3,10)', '', '(2,3,10,11)', '', '(2,3,10,11,12,13)', '', 'All', '']
    ax.set_xticks(ind)
    ax.set_xticklabels(set_labels, ha="center") 
    handles = [mpatches.Patch(color = color_train, label = "Training"), mpatches.Patch(color = color_test, label = "Testing")]

    plt.figlegend(handles = handles, loc = 1, fontsize = 'x-large') 
    plt.grid(True)
    plt.xlabel("Sets of channels used for feature extraction", fontsize = 18)
    plt.ylabel("Accuracy (%)", fontsize = 18)
    labelsize_ticks = 15
    plt.tick_params(axis='both', which='major', labelsize=labelsize_ticks)
#    plt.savefig(title)
    plt.show()

#plot_results()
plot_knn_results()
