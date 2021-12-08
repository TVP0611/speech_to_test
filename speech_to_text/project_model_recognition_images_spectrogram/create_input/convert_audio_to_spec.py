import matplotlib
matplotlib.use('Agg')# fix error: fail to allocate bitmap
import librosa
import librosa.display
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import figure
import gc

def convert_audio2spec(path_file_audio, path_save_spec, name_spec):
    plt.interactive(False)
    if os.path.isdir(path_save_spec):
        pass
    else:
        os.mkdir(path_save_spec)
    print(name_spec)
    # Load data
    data, sample_rate = librosa.load(path=path_file_audio, sr=8000)
    # setup parameters of matplotlib
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    # feature extract audio
    S = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_fft=2048, hop_length=256)
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max))
    # save spectrogram
    filename = path_save_spec + '/' + '{0}.png'.format(name_spec)
    plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
    # close thread of matplotlib (affect memory of pc)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    # reset variable
    del path_file_audio, name_spec, data, sample_rate, fig, ax, S

path_data_audio = "D:/train_model_speech_to_test/speech_to_text/project_model_recognition_images_spectrogram/data_augment/"
path_save_spectrograms = "D:/train_model_speech_to_test/speech_to_text/project_model_recognition_images_spectrogram/data_spectrograms/"

# read file
directories = os.listdir(path_data_audio)
for files in directories:
    num_files = os.listdir(os.path.join(path_data_audio + files))[0:4000]
    for file in num_files:
        name = file.split(".wav")[0]
        convert_audio2spec(path_file_audio=os.path.join(path_data_audio + files + "/" + file), path_save_spec=os.path.join(path_save_spectrograms + files), name_spec=name)
