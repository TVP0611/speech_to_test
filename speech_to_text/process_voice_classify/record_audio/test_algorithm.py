# import threading
# import numpy as np
#
# lt_synthetic = []
#
# l1 = [[12, 4234, 213, 3544, 5435, 567, 443, 435, 232, 32, 21, 34, 431, 4432],
#       [11, 343, 412, 345, 633, 267, 22, 55, 11, 452, 244, 211, 41, 54],
#       [32, 345, 222, 662, 456, 22, 12, 45, 255, 432, 435, 523, 45, 55],
#       [3, 54, 2, 2, 42, 11, 33, 1, 0, 12, 12, 1, 6, 1],
#       [32, 4, 22, 45, 11, 12, 4, 56, 55, 7, 1, 2, 8, 2],
#       [12, 4234, 213, 3544, 5435, 567, 443, 435, 232, 32, 21, 34, 431, 4432],
#       [11, 343, 412, 345, 633, 267, 22, 55, 11, 452, 244, 211, 41, 54],
#       [32, 345, 222, 662, 456, 22, 12, 45, 255, 432, 435, 523, 45, 55],
#       [3, 54, 2, 2, 42, 11, 33, 1, 0, 12, 12, 1, 6, 1],
#       [32, 4, 22, 45, 11, 12, 4, 56, 55, 7, 1, 2, 8, 2]]
#
# # l2 = [11, 343, 412, 345, 633, 267, 22, 55, 11, 452, 244, 211, 41, 54]
# # l3 = [32, 345, 222, 662, 456, 22, 12, 45, 255, 432, 435, 523, 45, 55]
# # l4 = [3, 54, 2, 2, 42, 11, 33, 1, 0, 12, 12, 1, 6, 1]
# # l5 = [32, 4, 22, 45, 11, 12, 4, 56, 55, 7, 1, 2, 8, 2]
# # l6 = [12, 4234, 213, 3544, 5435, 567, 443, 435, 232, 32, 21, 34, 431, 4432]
# # l7 = [11, 343, 412, 345, 633, 267, 22, 55, 11, 452, 244, 211, 41, 54]
# # l8 = [32, 345, 222, 662, 456, 22, 12, 45, 255, 432, 435, 523, 45, 55]
# # l9 = [3, 54, 2, 2, 42, 11, 33, 1, 0, 12, 12, 1, 6, 1]
# # l10 = [32, 4, 22, 45, 11, 12, 4, 56, 55, 7, 1, 2, 8, 2]
#
# def add_list():
#     for i in l1:
#         if max(i) >= 200:
#             lt_synthetic.extend(i)
#
#
# t2 = threading.Thread(target=add_list, daemon=True)
# ######      Bắt đầu chạy thread
# t2.start()
#
# while True:
#     print(len(lt_synthetic))
#     if len(lt_synthetic) >= 85:
#         break
# print('done')

# from dvg_ringbuffer import RingBuffer
# import numpy as np
#
# rb = RingBuffer(5, dtype=np.int16)  # --> rb[:] = array([], dtype=int32)
# rb.extendleft([1, 2, 3, 4, 5])        # --> rb[:] = array([2, 3, 4, 5])
# # rb.popleft()
# rb.extendleft([6])
# print(rb)

import pyaudio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter
from itertools import chain
from filter_noise import append_silence, filter_audio
import threading
# from code_filter_and_normalize import butter_bandpass_filter, append_silence
from dvg_ringbuffer import RingBuffer
# from test_model import *
import time
import os
# kss = Keyword_Spotting_Service()
# kss1 = Keyword_Spotting_Service()

# # check that different instances of the keyword spotting service point back to the same object (singleton)
# assert kss is kss1

######      Khởi tạo giá trị khi record (tỉ lệ lấy mẫu = 8000 hz, channel = 1, giá trị tín hiệu theo kiểu = int16)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=8000,
                input=True, frames_per_buffer=1024)

rb = RingBuffer(640000, dtype=np.int16)
pt = 0
val_wait = 0
signals = []
low = 100.0
high = 3900.0
sr = 8000
file_save = 0
audio = []
audio_split = []
data_new = []
name = "noise_p"
path = "D:/train_model_speech_to_test/speech_to_text/process_voice_classify/audio_sample/"
file_save = 0

# max_val = []
# count_pulse = []

# def record():
#     global pt, val_wait, audio
#     while True:
#         #####      Bắt tín hiệu audio
#         data = np.frombuffer(stream.read(1024), dtype=np.int16)
#         # print(max(data))
#         ######     Nhận biết tiếng nói và Cắt câu
#         if max(data) >= 2000:
#             # data = normalize(data)
#             # data = filter_audio(sr, data)
#             data = data.astype(np.int16)
#             # ls.extend(data)
#             # rb.extend(data)
#             audio.extend(data)
#             val_wait = 0
#             # if len(audio) >= 40000:
#             #     audio = np.array(audio).astype(np.int16)
#             #     sf.write(
#             #         "D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/test_realtime/file_audio_rt_test.wav",
#             #         audio, sr, 'PCM_16')
#             #     print("save")
#             #     break
#         else:
#             val_wait += 1
#             print(val_wait)
#             if val_wait == 100:
#                 rb.clear()
#                 pt=0
#                 print("break")
#                 break
#
# record()
# l2 = [11, 343, 412, 345, 633, 267, 22, 55, 11, 452, 244, 211, 41, 54]
# rb.extend(l2)
# print(len(rb), pt)

def record():
    global pt, val_wait, audio
    while True:
        #####      Bắt tín hiệu audio
        data = np.frombuffer(stream.read(1024), dtype=np.int16)
        # print(max(data))
        ######     Nhận biết tiếng nói và Cắt câu
        if max(data) >= 1000:
            # data = normalize(data)
            # data = filter_audio(sr, data)
            data = data.astype(np.int16)
            # ls.extend(data)
            # rb.extend(data)
            rb.extend(data)
            val_wait = 0

        else:
            val_wait += 1
            # print(val_wait)
            if val_wait == 60:
                rb.clear()
                pt = 0


t2 = threading.Thread(target=record, daemon=True)
######      Bắt đầu chạy thread
t2.start()

while 1:
    # pass
    try:
        assert len(np.array(rb)) >= 1024
    except:
        pass
        # print("Audio length is too short or incorrect sampling frequency")
    else:
        signals = rb[pt: pt + 64]
        try:
            assert len(signals) >= 64
        except:
            pass
            # print("Signals length is too short")
        else:
            # count_pulse_down = []
            # len_max = []
            pt = pt + 64
            thresh_val = max(signals)
            # max_val.append(max(signals))
            if thresh_val >= 700:
                # count_pulse.append(1)
                data_new.append(signals)
            elif thresh_val < 700:  # and a == 0:
                # count_pulse.append(0)
                # len_max.append(len(data_new))
                if len(data_new) > 5:
                    data_new = list(chain.from_iterable(data_new))
                    data_new = append_silence(data_new)
                    # data_new = normalize_audio(fs=sr, signals=data_new)
                    # audio_split.append(data_new)
                    data_new1 = np.array(data_new)#.astype(np.int16)
                    # print(type(data_new))
                    # sf.write("D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/test_realtime/file_audio_rt_{0}.wav".format(file_save), data_new, sr, 'PCM_16')
                    # keyword, acc = kss.predict(data_new1)  # test_audio/bat/bat_42.wav
                    # print(keyword, max(acc) * 100)
                    data_new = np.array(data_new).astype(np.int16)
                    if os.path.isdir(path + name):
                        sf.write(path + name + "/" + "{0}_{1}.wav".format(name, file_save), data_new, sr, 'PCM_16')
                    else:
                        os.mkdir(path + name)
                        sf.write(path + name + "/" + "{0}_{1}.wav".format(name, file_save), data_new, sr, 'PCM_16')
                    # sf.write(path + "file_audio_rt_{0}.wav".format(file_save), data_new, sr, 'PCM_16')
                    data_new = []
                    file_save += 1
                    print('save done {}'.format(file_save))

