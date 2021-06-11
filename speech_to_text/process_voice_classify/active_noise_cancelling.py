# Importing libraries
import os

os.system('cls')
import numpy as np
import matplotlib.pyplot as plt
from math import pi
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def Convert(lst):
    # lst = np.array(lst)
    return -lst
# N = 145
# nvec = np.linspace(-N + 1, N - 1, num=N)
# y1 = np.cos((2.0 * pi * nvec / N))
# # clf()
# plt.plot(nvec, y1, label="Original")
# # plt.show()
# y1_reverse = Convert(y1)
# out_arr = np.add(y1, y1_reverse)
#
# # y1_reverse = np.cos(-(2.0 * pi * nvec / N))#.astype(np.int16)
# plt.plot(nvec, y1_reverse, label="Reversed")
# plt.xlabel('time')
# plt.ylabel('Signal')
# plt.legend(bbox_to_anchor=(.5, .9), loc=1, borderaxespad=0.)
# # plt.legend("Original","Time Reversed")
# plt.show()