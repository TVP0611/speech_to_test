import warnings
warnings.simplefilter("ignore", DeprecationWarning)
# import matplotlib
# matplotlib.use('Agg')
# import re
import pyaudio
from numpy.linalg import norm
import numpy as np
import threading
# import soundfile as sf
import soundfile as sf
from scipy.signal import butter, lfilter
# from scipy.io import wavfile
# from scipy import signal
from pyAudioAnalysis import audioTrainTest as aT
# # import time
# from pydub import AudioSegment
# import os.path
# import speech_recognition as src
# from datetime import datetime
# # import shutil
# import regex
# import matplotlib.pyplot as plt
# # from matplotlib import figure
# import librosa
# import librosa.display
# # import gc
# from skimage.metrics import structural_similarity
# import imutils
# import cv2
# import io
# import base64
# # from pyimagesearch.hashing import convert_hash
# # from pyimagesearch.hashing import dhash
# import pickle
# import time
# from skimage.metrics import structural_similarity
# import imutils
# import os
from dtw import dtw
from python_speech_features import mfcc
import pickle
from annoy import AnnoyIndex
from collections import Counter
from preprocess import *
from tensorflow.keras import models

######      Khởi tạo giá trị khi record (tỉ lệ lấy mẫu = 8000 hz, channel = 1, giá trị tín hiệu theo kiểu = int16)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=8000,
                input=True, frames_per_buffer=1024)

######      tạo array lưu giá trị audio
ls = []
ls_copy = []
ls_score_lb = []
ls_aver = []
ls_score = []
ls_result = []
save_file = 0
pt = 0
sr = 8000
low = 100.0
high = 3900.0

model = models.load_model('model.h5')

def predict(data, model):
    feature_dim_2 = 11
    feature_dim_1 = 20
    channel = 1
    sample = wav2mfcc_ram(data)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    n = np.max(model.predict(sample_reshaped))
    # m, k, h = get_labels()
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))], (n * 100)

def normalize(signal):
    a = -18000
    b = 18000
    # solving system of linear equations one can find the coefficients
    A = np.min(signal)
    B = np.max(signal)
    C = (a - b) / (A - B)
    k = (C * A - a) / C
    return (signal) * C

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
    return -lst

def append_silence(data_new):
    for x in range(int(1000)):
        data_new.insert(0, 0)
    if len(data_new) < 4800:
        for y in range(int(4800-len(data_new))):
            data_new.append(0)
    return data_new

def extract_features(y, sr=8000, nfilt=10, winsteps=0.02):
    try:
        feat = mfcc(y, sr, nfilt=nfilt, winstep=winsteps)
        return feat
    except:
        raise Exception("Extraction feature error")

def crop_feature(feat, i = 0, nb_step=10, maxlen=100):
    crop_feat = np.array(feat[i: i + nb_step]).flatten()
    # print(crop_feat.shape)
    crop_feat = np.pad(crop_feat, (0, maxlen - len(crop_feat)), mode='constant')
    return crop_feat

def google_translate():
    pass

def cat_tu():

    # features_crop = pickle.load(open('features_crop.pk', 'rb'))
    # songs_crop = pickle.load(open('songs_crop.pk', 'rb'))
    # features = pickle.load(open('features.pk', 'rb'))
    # songs = pickle.load(open('songs.pk', 'rb'))
    # f = 100
    # u = AnnoyIndex(f, 'angular')
    # u.load('music.ann')
    global ls, ls_copy, save_file, pt, sr
    a = 0
    while True:
        if len(ls) > 1024:
            n_data = ls[pt:pt + 1024]
            # print(type(n_data))
            # n_data = normalize(n_data)
            # print(len(n_data))
            if len(n_data) >= 1024:
                pt = pt + 1024
                n = 0
                for i in range(int(len(n_data) / 64)):
                    data_split = n_data[n: n + 64]
                    n = n + 64
                    # print(max(data_split))
                    if max(data_split) > 850:
                        ls_copy.extend(data_split)
                        a = 0

                    elif len(ls_copy) > 500 and a == 0:
                        # scale_val = 20000 / max(ls_copy)
                        # ls_copy = list(map(lambda x: x * scale_val, ls_copy))
                        ls_copy = append_silence(ls_copy)
                        ls_copy = np.array(ls_copy)
                        data_new = ls_copy.astype(np.float32)
                        ls_copy = ls_copy.astype(np.int16)

                        # # print(type(ls_copy[0]))
                        # c, p, p_nam = aT.file_classification_on_ram(ls_copy, 8000, "svm_classical_metal", "svm_rbf")
                        # # print(p)
                        # val_max = np.where(p == max(p))[0]
                        # if max(p)*100 >= 60:
                        #     print('P(svm)({0} = {1})'.format(str(p_nam[val_max[0]]), str(max(p))))
                        #
                        # feat1 = extract_features(ls_copy)
                        #
                        # results = []
                        # for j in range(0, feat1.shape[0], 1):
                        #     crop_feat = crop_feature(feat1, j, nb_step=10)
                        #     result = u.get_nns_by_vector(crop_feat, n=6)
                        #     result_songs = [songs_crop[k] for k in result]
                        #     results.append(result_songs)
                        #
                        # results = np.array(results).flatten()
                        #
                        # most_song = Counter(results)
                        # most_song.most_common()
                        # m = list(most_song.most_common())
                        # distance = []
                        # for j in features:
                        #     manhattan_distance = dist = lambda x, y: norm(x - y, ord=1)
                        #     d, cost_matrix, acc_cost_matrix, path = dtw(j, feat1, dist=manhattan_distance)
                        #     distance.append(d)
                        # min_dis = distance.index(min(distance))
                        # song = songs[min_dis]
                        # name = m[0][0]
                        # name = name.split('\\')[1].split('_')[1]
                        # song = song.split('\\')[1].split('_')[1]

                        # if min(distance) < 500 and m[0][1] >= 3:
                        # print(m[0])
                        # print(min(distance))
                        # print(song)
                        # if (name == song == 'tat' and min(distance) < 300):
                        #     print(min(distance))
                        #     print(song)

                        # if (name == song == 'bat' and min(distance) < 450) \
                        #         or (name == song == 'tat' and min(distance) < 500) \
                        #         or (name == song == 'ngat' and min(distance) < 400) \
                        #         or (name == song == 'on' and min(distance) < 650) \
                        #         or (name == song == 'off' and min(distance) < 650) \
                        #         or (name == song == 'dong' and min(distance) < 600) \
                        #         or (name == song == 'mo' and min(distance) < 2200):
                        #     print(min(distance))
                        #     print(song)

                        # if m[0][1] >= 8:
                        #     print(m[0])

                        label_name, acc = predict(data_new, model=model)
                        # acc_val = np.max(model.predict(name_predict))
                        # label_name, acc = get_labels()[0][np.argmax(model.predict(name_predict))], (acc_val * 100)
                        print(label_name, round(acc, 3))

                        sf.write('save_audio/test_{0}.wav'.format(save_file), ls_copy, 8000, 'PCM_16')
                        save_file += 1
                        print('save_file: {0}'.format(save_file))
                        ls_copy = []
                        # m = 0
                    else:
                        a += 1



######      Creating thread processing audio
t2 = threading.Thread(target=cat_tu, daemon= True)
######      Bắt đầu chạy thread
t2.start()

while True:
    #####      Bắt tín hiệu audio
    data = np.fromstring(stream.read(1024), dtype=np.int16)
    # print(max(data))
    ######     Nhận biết tiếng nói và Cắt câu
    if max(data) >= 2000:
        # data = normalize(data)
        data = butter_bandpass_filter(data, low, high, sr, 6)
        data = data.astype(np.int16)
        ls.extend(data)
        a = 0
        if len(ls) > 4800000:
            ls.clear()
            pt = 0
