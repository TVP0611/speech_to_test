from scipy.io import wavfile
import wave
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
import os

Path_raw_input = "raw_data_audio/"
Path_format_output = "data_wav/"
number_audio_save = 0


def format_wav(filepath, _name_command):
    global number_audio_save
    print(_name_command + "/" + filepath)
    song = AudioSegment.from_file(filepath)
    if song.channels == 2:
        song = song.split_to_mono()[0]
    song = song.set_frame_rate(8000)
    if os.path.isdir(Path_format_output + _name_command):
        song.export(
            out_f=Path_format_output + _name_command + "/" + "{0}_format_{1}.wav".format(_name_command, number_audio_save),
            format="wav")
        number_audio_save += 1
    else:
        os.mkdir(Path_format_output + _name_command)
        song.export(
            out_f=Path_format_output + _name_command + "/" + "{0}_format_{1}.wav".format(_name_command, number_audio_save),
            format="wav")
        number_audio_save += 1


fol_raw = os.listdir(Path_raw_input)
for _name_fol_raw in fol_raw:
    _audio_files = os.listdir(Path_raw_input + _name_fol_raw)
    for _file_audio in _audio_files:
        format_wav(filepath=Path_raw_input + _name_fol_raw + "/" + _file_audio,
                   _name_command=_name_fol_raw)
    number_audio_save = 0
