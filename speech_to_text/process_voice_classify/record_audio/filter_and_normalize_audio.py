from scipy.io import wavfile
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter
from itertools import chain
# from normalize_audio import normalize_audio

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

def append_silence(data_new):
    for x in range(int(3000)):
        data_new.insert(0, 0)
    if len(data_new) < 8000:
        for y in range(int(8000-len(data_new))):
            data_new.append(0)
    return data_new

def Convert(lst):
    return -lst

def filter_and_normalize(sr, data, local_array = None, file_save = None):
    try:
        len_signals = len(data)
        assert len_signals >= 1024
        assert sr >= 8000
    except:
        print("Audio length is too short or incorrect sampling frequency")
    else:
        audio_split = []
        low = 100.0
        high = 3900.0
        data_new = []
        max_val = []
        # len_max = []
        pt = 0
        a = 0
        # file_save = 0
        for i in range(int(len(data)/1024)):
            data_n = data[pt:pt+1024]
            data_n = butter_bandpass_filter(data_n, low, high, sr, 6)
            data_n = data_n.astype(np.int16)
            pt = pt + 1024
            n = 0
            for j in range(int(len(data_n) / 64)):
                data_scan = data_n[n:n+64]
                n = n + 64
                thresh_val = max(data_scan)
                # max_val.append(max(data_scan))
                if thresh_val >= 700:
                    data_new.append(data_scan)
                elif thresh_val < 700:  #and a == 0:
                    # len_max.append(len(data_new))
                    if len(data_new) > 5:
                        data_new = list(chain.from_iterable(data_new))
                        data_new = append_silence(data_new)
                        data_new = normalize_audio(fs=sr, signals=data_new)
                        audio_split.append(data_new)
                        # print(type(data_new))
                        sf.write("save_audio_test/file_agc_{0}.wav".format(file_save), data_new, sr, 'PCM_16')
                        data_new = []
                        file_save += 1
                        print('save done {}'.format(file_save))
        #######        Show signal_split
        # max_val = np.array(max_val)
        # # len_max = np.array(len_max)
        # plt.plot(max_val)
        # plt.show()
        print('done')
        if local_array == None:
            pass
        else:
            local_array += 1024
        return audio_split, local_array, file_save

def filter_and_normalize_on_ram(sr, data, local_array = None, file_save = None):
    try:
        len_signals = len(data)
        assert len_signals >= 64
        assert sr >= 8000
    except:
        print("Audio length is too short or incorrect sampling frequency")
    else:
        audio_split = []
        low = 100.0
        high = 3900.0
        data_new = []
        max_val = []
        count_pulse = []
        # count_pulse_down = []
        # len_max = []
        pt = 0
        a = 0
        # file_save = 0
        # for i in range(int(len(data)/1024)):
        #     data_n = data[pt:pt+1024]
        #     data_n = butter_bandpass_filter(data_n, low, high, sr, 6)
        #     data_n = data_n.astype(np.int16)
        #     pt = pt + 1024
        n = 0
        for j in range(int(len(data) / 64)):
            data_scan = data[n:n+64]
            n = n + 64
            thresh_val = max(data_scan)
            max_val.append(max(data_scan))
            if thresh_val >= 700:
                count_pulse.append(1)
                data_new.append(data_scan)
            elif thresh_val < 700:  #and a == 0:
                count_pulse.append(0)
                # len_max.append(len(data_new))
                if len(data_new) > 5:
                    data_new = list(chain.from_iterable(data_new))
                    data_new = append_silence(data_new)
                    data_new = normalize_audio(fs=sr, signals=data_new)
                    audio_split.append(data_new)
                    # print(type(data_new))
                    # sf.write("save_audio_test/file_agc_{0}.wav".format(file_save), data_new, sr, 'PCM_16')
                    data_new = []
                    file_save += 1
                    # print('save done {}'.format(file_save))
                    if local_array == None:
                        pass
                    else:
                        local_array += 1024
                    return audio_split, local_array, file_save

            count_element = count_pulse.count(1)
            if count_pulse[-1] != 0 and count_element > 3:
                # data_new = append_silence(data_new)
                audio_split.append(data_new)
                data_new = []
                file_save = None
                if local_array == None:
                    pass
                else:
                    local_array += 1024
                return audio_split, local_array, file_save
        #######        Show signal_split
        # max_val = np.array(max_val)
        # # len_max = np.array(len_max)
        # plt.plot(max_val)
        # plt.show()
        # print('done')
        # if local_array == None:
        #     pass
        # else:
        #     local_array += 1024
        # return audio_split, local_array, file_save


if __name__ == '__main__':
    sr, signal = wavfile.read('test.wav')
    low = 100.0
    high = 3900.0
    data = []
    data_filter = Convert(signal[:1000])
    data_split = np.array_split(signal, int(len(signal)/(1000-10)))
    for i in data_split:
        data_filter = data_filter[:len(i)]
        y = np.add(data_filter, i)
        data.extend(y)

    data = butter_bandpass_filter(data, low, high, sr, 6)
    data = data.astype(np.int16)
    data = normalize_audio(sr, data)
    audio_split = filter_and_normalize(sr, data, local_array=1024)
    print("done")