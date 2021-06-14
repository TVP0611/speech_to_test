from scipy.io.wavfile import read, write
import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf
from functools import reduce
import os

def append_silence(data_new):
    for x in range(int(3000)):
        data_new.insert(0, 0)
    if len(data_new) < 8000:
        for y in range(int(8000-len(data_new))):
            data_new.append(0)
    return data_new

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

def normalize(infile, rms_level=0):
    """
    Normalize the signal given a certain technique (peak or rms).
    Args:
        - infile    (str) : input filename/path.
        - rms_level (int) : rms level in dB.
    """
    # read input file
    fs, sig = read(filename=infile)

    # linear rms level and scaling factor
    r = 10**(rms_level / 10.0)
    a = np.sqrt((len(sig) * (r**2)) / np.sum(sig**2))

    # normalize
    y = sig * a
    y = y.astype(np.int16)

    # construct file names
    output_file_path = os.path.dirname(infile)
    name_attribute = "output_file.wav"

    # export data to file
    # sf.write("D:/Project/process_voice/process_voice_classify/audio_data/bat/example.wav", y, fs, 'PCM_16')#.astype(np.int16)
name = "bat_p"
path = "D:/train_model_speech_to_test/speech_to_text/process_voice_classify/audio_sample/"
file_save = 0
# normalize(path, 20)
pt = 0
sr, data = read(path + name + ".wav")
data_new = []
a = 0
low = 100.0
high = 3900.0
data = butter_bandpass_filter(data, low, high, sr, 6)
data_n = data.astype(np.int16)
# data_n = [i * 12 for i in data]
# data_n = np.array(data_n)
# data_n = data_n.astype(np.int16)
# sf.write("test_audio/bat/bat_test.wav", data_n, sr, 'PCM_16')
# print("done")
for l in range(int(len(data_n) / 1024)):
    data1 = data_n[pt:pt + 1024]
    # data1 = butter_bandpass_filter(data1, low, high, sr, 6)
    # data1 = data1.astype(np.int16)
    pt = pt + 1024
    n = 0
    for m in range(int(len(data1) / 64)):
        data_scan = data1[n:n + 64]
        n = n + 64
        # print(max(data_scan))
        if max(data_scan) > 800:
            y = data_scan
            data_new.extend(y)
            a = 0
        # else:
        #     data_filter = Convert(data_scan)
        #     y = np.add(data_filter, data_scan)
        #     data_new.extend(y)
        elif len(data_new) > 1000 and a == 0:
            # if max(data_new) < 22000:
            #     # value_thresh = int((22000 / max(data_new)))
            # data_new1 = [i * 10 for i in data_new]

            data_new = append_silence(data_new)
            data_new = np.array(data_new)
            data_new = data_new.astype(np.int16)

            if os.path.isdir('test_audio/test_sentence/' + name):
                sf.write("test_audio/test_sentence/{0}/{0}_{1}.wav".format(name, file_save), data_new, sr, 'PCM_16')
            else:
                os.mkdir('test_audio/test_sentence/' + name)
                sf.write("test_audio/test_sentence/{0}/{0}_{1}.wav".format(name, file_save), data_new, sr, 'PCM_16')

            # sf.write("test_audio/{0}/{0}_{1}.wav".format(name, file_save), data_new, sr, 'PCM_16')

            # data_new = data_new.astype(np.int16)
            # sf.write("file_mo_0/file_{0}_{1}.wav".format('bat', file_save), data_new, sr, 'PCM_16')
            data_new = []
            file_save += 1
            print('save done {}'.format(file_save))
            # else:
            #     data_new = append_silence(data_new)
            #     data_new = np.array(data_new)
            #     data_new = data_new.astype(np.int16)
            #     sf.write("audio_data/{0}_{1}.wav".format('bat', file_save), data_new, sr, 'PCM_16')

            #     # data_new = data_new.astype(np.int16)
            #     # sf.write("file_mo_0/file_{0}_{1}.wav".format('bat', file_save), data_new, sr, 'PCM_16')
            #     data_new = []
            #     file_save += 1
            #     print('save done {}'.format(file_save))
        else:
            a += 1
