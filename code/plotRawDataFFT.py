# Compute the FFT

import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack

def fft_computation(y):
	N = 384
	T = 1.0 / 128.0
	x = np.linspace(0.0, N * T, N)
	#y = data_array[:,2] - 2000
	yf = scipy.fftpack.fft(y)
	xf = np.linspace(0.0, 1.0 / (2.0 * T), N/2)

	plt.figure()
	plt.plot(xf, 2.0 / N * np.abs( yf[:N//2] ))
	plt.ylim((0, 600))
	plt.show(block=False)

basePath = 'raw_data/'
fileName = 'data_v_bb_2.npy'
data_array = np.load(basePath + fileName)

fft_computation(data_array[:, 11] - 2000)

fileName = 'data_v_bl_2.npy'
data_array = np.load(basePath + fileName)

fft_computation(data_array[:, 11] - 2000)

input()



