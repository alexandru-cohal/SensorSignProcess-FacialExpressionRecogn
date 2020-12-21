import numpy as np
import math
import matplotlib.pyplot as plt
import time

t = np.linspace(0, 1000, num=100000)
x = np.sin(t)

plt.ion()
t = -1
xArray = np.zeros((10,1))
while 1:
	t = t + 1
	xValue = np.sin(t)
	xArray = np.roll(xArray, 9)
	xArray[9] = xValue
	plt.plot(xArray)
	plt.pause(0.1)
	plt.cla()
	

