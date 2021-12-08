import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
import tensorflow.keras

from datetime import datetime
import soundfile as sf
import librosa
import numpy as np
import matplotlib.pyplot as plt

import pyaudio
from queue import Queue
from threading import Thread
import sys
import time

model = load_model('model.h5')

# # def get_audio_input_stream(callback):
# #     stream = pyaudio.PyAudio().open(
# #         format=pyaudio.paInt16,
# #         channels=1,
# #         rate=fs,
# #         input=True,
# #         frames_per_buffer=chunk_samples,
# #         input_device_index=0,
# #         stream_callback=callback)
# #     return stream
# #
# # def callback(in_data, frame_count, time_info, status):
# #     global run, timeout, data, silence_threshold
# #     data0 = np.frombuffer(in_data, dtype='int16')
# #     data = np.append(data, data0)
# #     if len(data) > feed_samples:
# #         data = data[-feed_samples:]
# #         q.put(data)
# #     return (in_data, pyaudio.paContinue)
# #
# #
# # while True:
# #         data = q.get()
# #         save_audio(data, "temp/temp.wav")
#
# model = load_model('model.h5')
# feature = extract_feature("audio_split/p_6.wav")
# x_t = np.zeros((1, feature.shape[0], feature.shape[1]))
# x_t[0, :, :] = feature
# r = model.predict(x_t)
# print(r)
# if r[0][0] > 0.5:
#     print("trigger word detected!")
#     print(r[0][0])
#
#
# # time.sleep(0.001)

def detect_triggerword_spectrum(x):
    """
    Function to predict the location of the trigger word.

    Argument:
    x -- spectrum of shape (freqs, Tx)
    i.e. (Number of frequencies, The number time steps)

    Returns:
    predictions -- flattened numpy array to shape (number of output time steps)
    """
    # the spectogram outputs  and we want (Tx, freqs) to input into the model
    # x = x.swapaxes(0, 1)
    # x = np.expand_dims(x, axis=0)
    x = x.reshape((1, x.shape[0], x.shape[1]))
    predictions = model.predict(x)
    return predictions.reshape(-1)


def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.5):
    """
    Function to detect new trigger word in the latest chunk of input audio.
    It is looking for the rising edge of the predictions data belongs to the
    last/latest chunk.

    Argument:
    predictions -- predicted labels from model
    chunk_duration -- time in second of a chunk
    feed_duration -- time in second of the input to model
    threshold -- threshold for probability above a certain to be considered positive

    Returns:
    True if new trigger word detected in the latest chunk
    """
    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False

chunk_duration = 1 # Each read length in seconds from mic.
fs = 8000 # sampling rate for mic
chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 10
# feed_samples = int(fs * feed_duration)
feed_samples = 8000

assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)

def get_spectrogram(data):
    """
    Function to compute a spectrogram.

    Argument:
    predictions -- one channel / dual channel audio data as numpy array

    Returns:
        pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
    """
    # nfft = 200  # Length of each window segment
    # fs = 8000  # Sampling frequencies
    # noverlap = 120  # Overlap between windows
    # nchannels = data.ndim
    # if nchannels == 1:
    #     pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap=noverlap)
    # elif nchannels == 2:
    #     pxx, _, _ = mlab.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
    # return pxx

    data = data.astype(np.float32)
    sr = 8000
    num_step = int(sr * 1)
    if len(data) > num_step:
        data = data[0: num_step]
        pad_width = 0
    else:
        pad_width = num_step - len(data)
    data = np.pad(data, (0, int(pad_width)), mode='constant')
    D = np.abs(librosa.core.stft(y=data, n_fft=2048)) ** 2
    S = librosa.feature.melspectrogram(S=D, sr=sr, n_fft=256)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB.T


def plt_spectrogram(data):
    """
    Function to compute and plot a spectrogram.

    Argument:
    predictions -- one channel / dual channel audio data as numpy array

    Returns:
    pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
    """
    nfft = 200  # Length of each window segment
    fs = 8000  # Sampling frequencies
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _, _ = plt.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, _, _, _ = plt.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
    return pxx

def get_audio_input_stream(callback):
    stream = pyaudio.PyAudio().open(
        format=pyaudio.paInt16,
        channels=1,
        rate=fs,
        input=True,
        frames_per_buffer=chunk_samples,
        input_device_index=0,
        stream_callback=callback)
    return stream

# Queue to communiate between the audio callback and main thread
q = Queue()

run = True

silence_threshold = 200

# Run the demo for a timeout seconds
timeout = time.time() + 1 * 60  # 0.5 minutes from now

# Data buffer for the input wavform
data = np.zeros(feed_samples, dtype='int16')


def callback(in_data, frame_count, time_info, status):
    global run, timeout, data, silence_threshold
    if time.time() > timeout:
        run = False
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        sys.stdout.write('-')
        return (in_data, pyaudio.paContinue)
    else:
        sys.stdout.write('.')
    data = np.append(data, data0)
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        # Process data async by sending a queue.
        q.put(data)
    return (in_data, pyaudio.paContinue)


stream = get_audio_input_stream(callback)
stream.start_stream()

# try:
while 1:
    now = datetime.now()
    data = q.get()
    # print(len(data))
    spectrum = get_spectrogram(data)
    preds = detect_triggerword_spectrum(spectrum)
    # print(preds[0])
    # new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
    if preds[0] > 0.7:
        sys.stdout.write('1')
        dt_string = now.strftime("%d%m%Y_%H%M%S")
        sf.write('user_data/id_user_{0}.wav'.format(dt_string), data, 8000, 'PCM_16')
        ######     save audio, print (+)
        sys.stdout.write('+')
# except (KeyboardInterrupt, SystemExit):
#     stream.start_stream()
#     stream.stop_stream()
#     stream.close()
#     timeout = time.time()
#     run = False
#
# stream.stop_stream()
# stream.close()


