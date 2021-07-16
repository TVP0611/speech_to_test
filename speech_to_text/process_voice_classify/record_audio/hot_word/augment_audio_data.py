import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pydub import AudioSegment
import os


def plot_spec(data: np.array, sr: int, title: str, fpath: str) -> None:
    label = str(fpath).split('/')[-1].split('_')[0]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].title.set_text(f'{title} / Label: {label}')
    ax[0].specgram(data, Fs=2)
    fig.savefig('temp1.png', bbox_inches="tight", pad_inches=0)
    ax[1].set_ylabel('Amplitude')
    ax[1].plot(np.linspace(0, 1, len(data)), data)
    # plt.show()


def add_noise(_wav, _num):
    ''' Add Noise
    Noise addition using normal distribution with mean = 0 and std =1
    Permissible noise factor value = x > 0.004
    '''
    for i in [0.004, 0.006, 0.008, 0.01]:
        _wav_n = wav + i * np.random.normal(0, 1, len(_wav))
        sf.write('positive/acis_{0}_noise_{1}.wav'.format(_num, i), _wav_n, 8000, 'PCM_16')


# def time_stretching(_wav, _num):
#     ''' Time-stretching the wave
#     Permissible factor values = 0.7 < x < 1.5
#     '''
#     _factor = [0.7, 0.9, 1.1]
#     for i in _factor:
#         _wav_time_stch = librosa.effects.time_stretch(_wav, i)
#         # plot_spec(data=wav_time_stch,sr=sr,title=f'Stretching the time by {factor}',fpath=file_path)
#         sf.write('positive/acis_{0}_time_stch_{1}.wav'.format(_num, i), _wav_time_stch, 8000, 'PCM_16')


def change_volume(_file_path, _num):
    ''' Change volume audio
    Up value: 3 <= x <= 15
    Down value: -10 < x < -2
    '''
    _song = AudioSegment.from_wav(_file_path)
    # boost volume
    _val_up = [3, 6, 9, 15, 20]
    for i in _val_up:
        _louder_song = _song + i
        _louder_song.export("positive/acis_{0}_up_vol_{1}.wav".format(_num, i), format='wav')
    # reduce volume
    _val_down = [-10, -8, -6, -4]
    for j in _val_down:
        _quieter_song = _song + j
        _quieter_song.export("positive/acis_{0}_down_vol_{1}.wav".format(_num, (j * (-1))), format='wav')


def pitch_shifting(_wav, _num):
    ''' pitch shifting of wav
    Permissible factor values = -5 <= x <= 5
    '''
    _n_steps = [-5, -3, -1, 1, 3, 5]
    for i in _n_steps:
        _wav_pitch_sf = librosa.effects.pitch_shift(_wav, sr, i)
        # plot_spec(data=wav_pitch_sf,sr=sr,title=f'Pitch shifting by {-5} steps',fpath=file_path)
        sf.write('positive/acis_{0}_pitch_sf_{1}.wav'.format(_num, i), _wav_pitch_sf, 8000, 'PCM_16')


path = "audio_split/"
file = os.listdir(path)
for f in file:
    wav, sr = librosa.load(path + f, sr=None)
    num_audio = f.split(".")[0]
    # plot_spec(data=wav, sr=sr, title='Original wave file', fpath=file_path)
    sf.write('positive/acis_{0}_original.wav'.format(num_audio), wav, 8000, 'PCM_16')
    add_noise(wav, num_audio)
    # time_stretching(wav, num_audio)
    pitch_shifting(wav, num_audio)
    change_volume(path + f, num_audio)

print("done")
