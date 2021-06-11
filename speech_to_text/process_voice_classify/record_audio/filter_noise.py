from scipy.io import wavfile
import soundfile as sf
import numpy as np
from scipy.signal import butter, filtfilt

def append_silence(data_new):
    for x in range(int(3000)):
        data_new.insert(0, 0)
    if len(data_new) < 8000:
        for y in range(int(8000-len(data_new))):
            data_new.append(0)
    return data_new

def filter_audio(sr:int, signal, thresh=6000):
    '''signal : int16, thresh(default) : 6000'''
    # check the volume loud or low (normalize (1))
    if max(signal) > 20000:
        down_vol = max(signal) // thresh
        signal = [s / down_vol for s in signal]
    elif max(signal) <= 6000:
        up_vol = 20000 // max(signal)
        signal = [s * up_vol for s in signal]
    # filter noise
    fc = 100  # Cut-off frequency of the filter
    w = fc / (sr / 2) # Normalize the frequency
    b = butter(5, w, 'high')
    output = filtfilt(b[0], b[1], signal).astype(np.int16)
    return output

if __name__ == '__main__':

    sr, signal = wavfile.read('D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/test_sentence/file_test_audio.wav')#D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/test_sentence/noise.wav
    output = filter_audio(sr, signal, 10000)
    sf.write("D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/test_sentence/file_test_audio.wav", output, sr, 'PCM_16')
    print("done")