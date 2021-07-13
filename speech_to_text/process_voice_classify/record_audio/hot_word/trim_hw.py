import pyaudio
import soundfile as sf
import numpy as np
from array import array
from itertools import chain
import threading
from dvg_ringbuffer import RingBuffer
import os
# from hotword_detection import wordRecorder as wr
from hotword_detection import hwDetector as hd
from scipy.io import wavfile

######      Khởi tạo giá trị khi record (tỉ lệ lấy mẫu = 8000 hz, channel = 1, giá trị tín hiệu theo kiểu = int16)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=8000,
                input=True, frames_per_buffer=1024)

# rb = RingBuffer(20, dtype=np.int16)
Data_hw = []
list_silence = []
hwDet = hd.hwDetector()
name = 'test_trim_hw'
file_save = 0
sr = 8000
count_silence = 0
position_data = []
position_silence = []
flag = False

def Average(lst):
    lst = list(map(abs, lst))
    return sum(lst) / len(lst)

def normalize(data):
    """

    This function is used to normalize the sampled audio stream such that all values lie in the range -16383 to 16384. This is because we use a 16-bit representation to store audio. Out of these 16 bits 1 bit is reserved as a sign bit.

    :param data: Recorded audio
    :type data: array
    :returns: Normalized audio
    :rtype: array

    """
    maxShort = 16384
    scale = float(maxShort) / max(abs(_data) for i in data)

    r = array('h')
    for i in data:
        r.append(int(i * scale))
    return r

def record_hw():
    global Data_hw
    while True:
        #####      Bắt tín hiệu audio
        _data = np.frombuffer(stream.read(1024), dtype=np.int16)
        # rb.append(_data)


# t2 = threading.Thread(target=record_hw(), daemon=True)
# ######      Bắt đầu chạy thread record
# t2.start()
print("Start")
while True:
    _data = np.frombuffer(stream.read(1024), dtype=np.int16)
    if max(_data) >= 750:
        flag = True
    if flag:
        Data_hw.extend(_data)
        if max(_data) < 750:
            count_silence += 1
            if count_silence >= 2:
                count_silence = 0
                print(Average(Data_hw))
                if len(Data_hw) >= 6700 and Average(Data_hw) > 400:
                    print(hwDet.isHotword_in_ram(Data_hw))
                    Data_hw = np.array(Data_hw)
                    sf.write('acis/' + "{0}_{1}.wav".format("trim_hw", file_save), Data_hw, 8000, 'PCM_16')
                    Data_hw = []
                    file_save += 1
                    print(file_save)
                    flag = False
                if len(Data_hw) >= 6700 and Average(Data_hw) < 400:
                    Data_hw.clear()

