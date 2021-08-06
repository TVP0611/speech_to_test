from scipy.io.wavfile import read, write
import numpy as np
from scipy.signal import butter, lfilter
import soundfile as sf
from functools import reduce
import os

Path_fol_format = "data_wav/"
Path_out_audio_splitting = "dataset_audio/"

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

file_save = 0
folders_format = os.listdir(Path_fol_format)
for _name_fol_format in folders_format:
    _audio_format_files = os.listdir(Path_fol_format + _name_fol_format)

    for _file_audio in _audio_format_files:
        # name = "bat"
        # path = "data_wav/bat/bat_format_0.wav"
        if _name_fol_format == "tat":
            print("#############" + Path_fol_format + _name_fol_format + "/" + _file_audio)
            # file_save = 0
            # normalize(path, 20)
            pt = 0
            sr, data = read(Path_fol_format + _name_fol_format + "/" + _file_audio) #+ name + ".wav")
            data_new = []
            a = 0
            # low = 60.0
            # high = 3900.0
            # data = butter_bandpass_filter(data, low, high, sr, 5)
            # data_n = data.astype(np.int16)
            # # data_n = [i * 12 for i in data]
            # # data_n = np.array(data_n)
            # data_n = data_n.astype(np.int16)
            # sf.write("bat_test.wav", data_n, sr, 'PCM_16')
            # print("done")

            for l in range(int(len(data) / 1024)):
                data1 = data[pt:pt + 1024]
                # data1 = butter_bandpass_filter(data1, low, high, sr, 6)
                # data1 = data1.astype(np.int16)
                pt = pt + 1024
                n = 0
                for m in range(int(len(data1) / 64)):
                    data_scan = data1[n:n + 64]
                    n = n + 64
                    # print(max(data_scan))
                    if max(data_scan) > 600:
                        y = data_scan
                        data_new.extend(y)
                        a = 0
                    # else:
                    #     data_filter = Convert(data_scan)
                    #     y = np.add(data_filter, data_scan)
                    #     data_new.extend(y)
                    elif len(data_new) > 800 and a == 0:
                        # if max(data_new) < 22000:
                        #     # value_thresh = int((22000 / max(data_new)))
                        # data_new1 = [i * 10 for i in data_new]

                        data_new = append_silence(data_new)
                        data_new = np.array(data_new)
                        data_new = data_new.astype(np.int16)

                        if os.path.isdir(Path_out_audio_splitting + _name_fol_format):
                            sf.write(file=Path_out_audio_splitting + "{0}/{0}_{1}.wav".format(_name_fol_format, file_save), data=data_new, samplerate=sr, subtype='PCM_16')
                            print(Path_out_audio_splitting + "{0}/{0}_{1}.wav".format(_name_fol_format, file_save))
                        else:
                            os.mkdir(Path_out_audio_splitting + _name_fol_format)
                            sf.write(file=Path_out_audio_splitting + "{0}/{0}_{1}.wav".format(_name_fol_format, file_save), data=data_new, samplerate=sr, subtype='PCM_16')
                            print(Path_out_audio_splitting + "{0}/{0}_{1}.wav".format(_name_fol_format, file_save))
                        data_new = []
                        file_save += 1
                        # print('save done {}'.format(file_save))
                        # else:
                        #     data_new = append_silence(data_new)
                        #     data_new = np.array(data_new)
                        #     data_new = data_new.astype(np.int16)
                        #     sf.write("audio_data/{0}_{1}.wav".format('bat', file_save), data_new, sr, 'PCM_16')

                        #     # data_new = data_new.astype(np.int16)
                        #     # sf.write("file_mo_0/file_{0}_{1}.wav".format('bat', file_save), data_new, sr, 'PCM_16')
                        #     data_new = []
                        #     file_save += 1
                        # print('save done {}'.format(file_save))
                    else:
                        a += 1
    file_save = 0

print("done")
