import librosa
import numpy as np
import os

def extract_feature(path_file):
    """ convert audio to melspectrogram using librosa"""
    y, sr = librosa.load(path=path_file, sr=8000) # load audio
    sr = 8000
    num_step = int(sr * 1)
    if len(y) > num_step:
        y = y[0: num_step]
        pad_width = 0
    else:
        pad_width = num_step - len(y)
    y = np.pad(y, (0, int(pad_width)), mode='constant')
    D = np.abs(librosa.core.stft(y=y, n_fft=2048)) ** 2
    S = librosa.feature.melspectrogram(S=D, sr=sr, n_fft=256)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB.T







