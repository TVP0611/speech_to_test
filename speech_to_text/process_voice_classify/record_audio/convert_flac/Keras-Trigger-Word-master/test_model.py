# import numpy as np
# from pydub import AudioSegment
# import random
# import sys
# import io
# import os
# import glob
# import IPython
# from td_utils import *
# import matplotlib.pyplot as plt
#
# Tx = 5511 # The number of time steps input to the model from the spectrogram
# n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram
# Ty = 1375 # The number of time steps in the output of our model
# # Load audio segments using pydub
# activates, negatives, backgrounds = load_raw_audio()
#
# print("background len should be 10,000, since it is a 10 sec clip\n" + str(len(backgrounds[0])),"\n")
# print("activate[0] len may be around 1000, since an `activate` audio clip is usually around 1 second (but varies a lot) \n" + str(len(activates[0])),"\n")
# print("activate[1] len: different `activate` clips can have different lengths\n" + str(len(activates[1])),"\n")
#
#
# def get_random_time_segment(segment_ms):
#     """
#     Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
#
#     Arguments:
#     segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
#
#     Returns:
#     segment_time -- a tuple of (segment_start, segment_end) in ms
#     """
#
#     segment_start = np.random.randint(low=0,
#                                       high=10000 - segment_ms)  # Make sure segment doesn't run past the 10sec background
#     segment_end = segment_start + segment_ms - 1
#
#     return (segment_start, segment_end)
#
#
# def is_overlapping(segment_time, previous_segments):
#     """
#     Checks if the time of a segment overlaps with the times of existing segments.
#
#     Arguments:
#     segment_time -- a tuple of (segment_start, segment_end) for the new segment
#     previous_segments -- a list of tuples of (segment_start, segment_end) for the existing segments
#
#     Returns:
#     True if the time segment overlaps with any of the existing segments, False otherwise
#     """
#
#     segment_start, segment_end = segment_time
#
#     # Initialize overlap as a "False" flag.
#     overlap = False
#
#     # loop over the previous_segments start and end times.
#     # Compare start/end times and set the flag to True if there is an overlap
#     for previous_start, previous_end in previous_segments:
#         if segment_start <= previous_end and segment_end >= previous_start:
#             overlap = True
#
#     return overlap
#
#
# def insert_audio_clip(background, audio_clip, previous_segments):
#     """
#     Insert a new audio segment over the background noise at a random time step, ensuring that the
#     audio segment does not overlap with existing segments.
#
#     Arguments:
#     background -- a 10 second background audio recording.
#     audio_clip -- the audio clip to be inserted/overlaid.
#     previous_segments -- times where audio segments have already been placed
#
#     Returns:
#     new_background -- the updated background audio
#     """
#
#     # Get the duration of the audio clip in ms
#     segment_ms = len(audio_clip)
#
#     # Use one of the helper functions to pick a random time segment onto which to insert
#     # the new audio clip
#     segment_time = get_random_time_segment(segment_ms)
#
#     # Check if the new segment_time overlaps with one of the previous_segments. If so, keep
#     # picking new segment_time at random until it doesn't overlap.
#     while is_overlapping(segment_time, previous_segments):
#         segment_time = get_random_time_segment(segment_ms)
#
#     # Append the new segment_time to the list of previous_segments
#     previous_segments.append(segment_time)
#
#     # Superpose audio segment and background
#     new_background = background.overlay(audio_clip, position=segment_time[0])
#
#     return new_background, segment_time
#
#
# def insert_ones(y, segment_end_ms):
#     """
#     Update the label vector y. The labels of the 50 output steps strictly after the end of the segment
#     should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
#     50 following labels should be ones.
#
#
#     Arguments:
#     y -- numpy array of shape (1, Ty), the labels of the training example
#     segment_end_ms -- the end time of the segment in ms
#
#     Returns:
#     y -- updated labels
#     """
#
#     # duration of the background (in terms of spectrogram time-steps)
#     segment_end_y = int(segment_end_ms * Ty / 10000.0)
#
#     # Add 1 to the correct index in the background label (y)
#     for i in range(segment_end_y + 1, segment_end_y + 51):
#         if i < Ty:
#             y[0, i] = 1
#     return y
#
#
# # arr1 = insert_ones(np.zeros((1, Ty)), 9700)
# # plt.plot(insert_ones(arr1, 4251)[0,:])
# # print("sanity checks:", arr1[0][1333], arr1[0][634], arr1[0][635])
# # plt.show()
#
# def create_training_example(background, activates, negatives):
#     """
#     Creates a training example with a given background, activates, and negatives.
#
#     Arguments:
#     background -- a 10 second background audio recording
#     activates -- a list of audio segments of the word "activate"
#     negatives -- a list of audio segments of random words that are not "activate"
#
#     Returns:
#     x -- the spectrogram of the training example
#     y -- the label at each time step of the spectrogram
#     """
#
#     # Set the random seed
#     np.random.seed(18)
#
#     # Make background quieter
#     background = background - 20
#
#     # Initialize y (label vector) of zeros
#     y = np.zeros((1, Ty))
#
#     # Initialize segment times as an empty list
#     previous_segments = []
#     ### END CODE HERE ###
#
#     # Select 0-4 random "activate" audio clips from the entire list of "activates" recordings
#     number_of_activates = np.random.randint(0, 5)
#     random_indices = np.random.randint(len(activates), size=number_of_activates)
#     random_activates = [activates[i] for i in random_indices]
#
#     # Loop over randomly selected "activate" clips and insert in background
#     for random_activate in random_activates:
#         # Insert the audio clip on the background
#         background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
#         # Retrieve segment_start and segment_end from segment_time
#         segment_start, segment_end = segment_time
#         # Insert labels in "y"
#         y = insert_ones(y, segment_end)
#
#     # Select 0-2 random negatives audio recordings from the entire list of "negatives" recordings
#     number_of_negatives = np.random.randint(0, 3)
#     random_indices = np.random.randint(len(negatives), size=number_of_negatives)
#     random_negatives = [negatives[i] for i in random_indices]
#
#     # Loop over randomly selected negative clips and insert in background
#     for random_negative in random_negatives:
#         # Insert the audio clip on the background
#         background, _ = insert_audio_clip(background, random_negative, previous_segments)
#
#     # Standardize the volume of the audio clip
#     background = match_target_amplitude(background, -20.0)
#
#     # Export new training example
#     file_handle = background.export("train" + ".wav", format="wav")
#     print("File (train.wav) was saved in your directory.")
#
#     # Get and plot spectrogram of the new recording (background with superposition of positive and negatives)
#     x = graph_spectrogram("train.wav")
#
#     return x, y
#
# x, y = create_training_example(backgrounds[0], activates, negatives)
#
# plt.plot(y[0])
# plt.show()
#

import pyaudio
from queue import Queue
from threading import Thread
import sys
import time

import numpy as np
import time
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
# To generate wav file from np array.
from scipy.io.wavfile import write

from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam

# Use 1101 for 2sec input audio
Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram


# Use 272 for 2sec input audio
Ty = 1375# The number of time steps in the output of our model


# GRADED FUNCTION: model

def model(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """

    X_input = Input(shape=input_shape)

    ### START CODE HERE ###

    # Step 1: CONV layer (≈4 lines)
    X = Conv1D(filters=196, kernel_size=15, strides=4)(X_input)  # CONV1D
    X = BatchNormalization()(X)  # Batch normalization
    X = Activation('relu')(X)  # ReLu activation
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 2: First GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization

    # Step 3: Second GRU Layer (≈4 lines)
    X = GRU(units=128, return_sequences=True)(X)  # GRU (use 128 units and return the sequences)
    X = Dropout(0.8)(X)  # dropout (use 0.8)
    X = BatchNormalization()(X)  # Batch normalization
    X = Dropout(0.8)(X)  # dropout (use 0.8)

    # Step 4: Time-distributed dense layer (≈1 line)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)  # time distributed  (sigmoid)

    ### END CODE HERE ###

    model = Model(inputs=X_input, outputs=X)

    return model

model = model(input_shape = (Tx, n_freq))
# model.summary()
opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

model = load_model('./models/tr_model1.h5')


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
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
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

chunk_duration = 0.5 # Each read length in seconds from mic.
fs = 44100 # sampling rate for mic
chunk_samples = int(fs * chunk_duration) # Each read length in number of samples.

# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_duration = 10
feed_samples = int(fs * feed_duration)

assert feed_duration/chunk_duration == int(feed_duration/chunk_duration)


def get_spectrogram(data):
    """
    Function to compute a spectrogram.

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
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
    return pxx


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

silence_threshold = 100

# Run the demo for a timeout seconds
timeout = time.time() + 0.5 * 60  # 0.5 minutes from now

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

try:
    while run:
        data = q.get()
        spectrum = get_spectrogram(data)
        preds = detect_triggerword_spectrum(spectrum)
        new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
        if new_trigger:
            sys.stdout.write('1')
except (KeyboardInterrupt, SystemExit):
    stream.stop_stream()
    stream.close()
    timeout = time.time()
    run = False

stream.stop_stream()
stream.close()





