import librosa
import numpy as np
import soundfile as sf

path_file = 'train_audio/background/raw_2.wav'
y, sr = librosa.load(path_file, sr=8000)

print(len(y))

array_audio_1s = []
position = 0
file_save = 778

while True:
    data = y[position:position+8000]
    position = (position + 8000)-4000
    # print(data)
    sf.write('negatives/neg_{0}.wav'.format(file_save), data, 8000, 'PCM_16')
    print(file_save)
    file_save += 1
    if position >= len(y):
        print('done')
        break

