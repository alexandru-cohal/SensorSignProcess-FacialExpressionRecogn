#from __future__ import division, print_function
import numpy as np
from numpy.random import randn
from numpy.fft import rfft
from scipy import signal
import matplotlib.pyplot as plt
import math

def low_filter_channel(data_array, channel, sampling_f = 128, cut_frequency = 16, filter_order = 15):
    low_f_cut_norm = cut_frequency/(2.0*sampling_f)
    low_f_cut_rad = 2.0 * low_f_cut_norm
    b, a = signal.butter(filter_order, low_f_cut_rad, analog=False)
    return (signal.filtfilt(b,a,data_array[:,channel]))


def low_filter_channel_all(data_array, sampling_f = 128, cut_frequency = 16, filter_order = 15):
    for channel_index in range(data_array.shape[1]):
        try:
            filtered_array

        except:
            filtered_array = low_filter_channel(data_array, channel_index, sampling_f, cut_frequency, filter_order) 

        else:
            filtered_array = np.vstack((filtered_array, low_filter_channel(data_array, channel_index, sampling_f, cut_frequency, filter_order)))

    return np.transpose(filtered_array)


def main():
    path = '/home/vitorroriz/sensorsig/rawdata_01-03-18/'
    file_name = 'v_bl_1.npy'

    data_array = np.load(path + file_name)
    neutral = np.load(path + 'v_neutral_0.npy')
    open_moutn = np.load(path + 'v_om_0.npy')
    neutral2 = np.load(path + 'v_neutral_1.npy')
    open_moutn2 = np.load(path + 'v_om_1.npy')
    blink_l1 = np.load(path + 'v_n_1.npy')
    blink_l2 = np.load(path + 'v_n_2.npy')

    print "data_array shape : " + str(data_array.shape)
    y = low_filter_channel_all(data_array)
    print "y shape : " + str(y.shape)

    fs = 128.0
    signal_f = low_filter_channel(data_array, 2)
    '''
    density = []
    for channel_index in range(14):
        f, Pxx_den = signal.welch(neutral[:,2], fs)
        ff, Pxx_denf = signal.welch(open_moutn[:,2], fs)
    '''
    my_scaling = 'spectrum'
    f_n0, den_n0 = signal.welch(neutral, fs,  axis=0, scaling=my_scaling)
    f_om0, den_om0 = signal.welch(open_moutn, fs,  axis=0, scaling=my_scaling)
    f_n1, den_n1 = signal.welch(neutral2, fs,  axis=0, scaling=my_scaling)
    f_om1, den_om1 = signal.welch(open_moutn2, fs,  axis=0, scaling=my_scaling)
    f_bl0, den_bl0 = signal.welch(blink_l1, fs,  axis=0, scaling=my_scaling)
    f_bl1, den_bl1 = signal.welch(blink_l2, fs,  axis=0, scaling=my_scaling)
    
    den_n0_all = den_n0.sum(axis=1) 
    den_n1_all = den_n1.sum(axis=1) 
    den_om0_all = den_om0.sum(axis=1)
    den_om1_all = den_om1.sum(axis=1)
    den_bl0_all = den_bl0.sum(axis=1)
    den_bl1_all = den_bl1.sum(axis=1)


    plt.semilogy(f_n0, den_n0_all, 'bx', f_om0, den_om0_all,'rx', f_n1, den_n1_all, 'gx', f_om1, den_om1_all, 'mx', f_bl0, den_bl0, 'kx', f_bl1, den_bl1, 'yx')
    plt.show()

