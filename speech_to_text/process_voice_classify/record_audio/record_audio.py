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
from test_model import *
import time

kss = Keyword_Spotting_Service()
kss1 = Keyword_Spotting_Service()

# check that different instances of the keyword spotting service point back to the same object (singleton)
assert kss is kss1

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
# max_val = []
# count_pulse = []

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
                    keyword, acc = kss.predict(data_new1)  # test_audio/bat/bat_42.wav
                    print(keyword, max(acc) * 100)
                    data_new = np.array(data_new).astype(np.int16)
                    sf.write("D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/test_realtime/file_audio_rt_{0}.wav".format(file_save), data_new, sr, 'PCM_16')
                    data_new = []
                    file_save += 1
                    print('save done {}'.format(file_save))