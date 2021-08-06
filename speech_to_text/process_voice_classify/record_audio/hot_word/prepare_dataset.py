import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)


import numpy as np
import librosa
import tqdm
import os
# import time

def extract_feature(path_file):
    y, sr = librosa.load(path_file, sr=8000)
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

if __name__ == '__main__':
    path_positive = "positive/"
    path_negative = "negatives/"

    sample_data = extract_feature("train_audio/background/1.wav")
    num_sample = len(os.listdir(path_negative)) + len(os.listdir(path_positive))

    x = np.zeros((num_sample, sample_data.shape[0], sample_data.shape[1]))
    y = np.zeros((num_sample, 2))

    i = 0
    for file_name in os.listdir(path_positive):
        feature = extract_feature(path_positive + file_name)
        x[i, :, :] = feature
        y[i, :] = np.array([1, 0])
        i += 1

    for file_name in os.listdir(path_negative):
        feature = extract_feature(path_negative + file_name)
        x[i, :, :] = feature
        y[i, :] = np.array([0, 1])
        i += 1

    print("Num sample: ", num_sample)
    # print("Time: ", time.time() - start)

    np.save("Xtrain.npy", np.asarray(x))
    np.save("Ytrain.npy", np.asarray(y))

print('done')