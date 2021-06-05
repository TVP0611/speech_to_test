from preprocess import *
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from scipy.io.wavfile import read, write

models = keras.models.load_model('model.h5')
feature_dim_2 = 11
feature_dim_1 = 20
channel = 1

def predict(data, model):
    sample = wav2mfcc(data)
    sample_reshaped = sample.reshape(1, feature_dim_1, feature_dim_2, channel)
    n = np.max(model.predict(sample_reshaped))
    m, k, h = get_labels()
    return get_labels()[0][
            np.argmax(model.predict(sample_reshaped))], (n * 100)

path = 'test_audio/'
entries = os.listdir(path)
for i in entries:
    name_predict, acc = predict(path + i, model=models)
    print(name_predict, round(acc, 3))

# sr, data = read(path + 'test_8.wav')
# data = data.astype(np.float32)
#
# name_predict, acc = predict(data, model=models)
# print(name_predict, round(acc, 3))