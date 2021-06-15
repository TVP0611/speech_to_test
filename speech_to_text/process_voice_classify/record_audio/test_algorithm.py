""" Code thu thập mẫu"""
# Cắt từ
# lưu data(.wav) theo folder đặt trên trước
import pyaudio
import soundfile as sf
import numpy as np
from itertools import chain
from filter_noise import append_silence, filter_audio
import threading
from dvg_ringbuffer import RingBuffer
import os

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
name = "noise_p" ####   Đặt tên foler ở đây
####    Vị trí lưu file
path = "D:/train_model_speech_to_test/speech_to_text/process_voice_classify/audio_sample/"
file_save = 0

def record():
    global pt, val_wait, audio
    while True:
        #####      Bắt tín hiệu audio
        data = np.frombuffer(stream.read(1024), dtype=np.int16)
        # print(max(data))
        ######     Nhận biết tiếng nói và Cắt câu
        if max(data) >= 1000:
            data = data.astype(np.int16)
            rb.extend(data)
            val_wait = 0
        else:
            val_wait += 1
            # print(val_wait)
            if val_wait == 60:
                rb.clear()
                pt = 0

t2 = threading.Thread(target=record, daemon=True)
######      Bắt đầu chạy thread record
t2.start()

while True:
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
            ####    Phát hiện giọng nói và cắt từ
            pt = pt + 64
            thresh_val = max(signals)
            if thresh_val >= 700:
                data_new.append(signals)
            elif thresh_val < 700:  # and a == 0:
                if len(data_new) > 5:
                    data_new = list(chain.from_iterable(data_new))
                    data_new = append_silence(data_new)
                    data_new1 = np.array(data_new)
                    data_new = np.array(data_new).astype(np.int16)
                    ####    Save file
                    if os.path.isdir(path + name):
                        sf.write(path + name + "/" + "{0}_{1}.wav".format(name, file_save), data_new, sr, 'PCM_16')
                    else:
                        os.mkdir(path + name)
                        sf.write(path + name + "/" + "{0}_{1}.wav".format(name, file_save), data_new, sr, 'PCM_16')
                    data_new = []
                    file_save += 1
                    print('save done {}'.format(file_save))

