import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

clip, sample_rate = librosa.load("D:/train_model_speech_to_test/speech_to_text/project_model_recognition_images_spectrogram/data_augment/bat/bat_121_pitch_sf_-1.wav", sr=8000)
fig = plt.figure(figsize=[0.72,0.72])
# fig, ax = plt.subplots()
ax = fig.add_subplot(111)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)
S = librosa.feature.melspectrogram(y=clip, sr=sample_rate, n_fft=2048, hop_length=256)
librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max))
# plt.show()
filename = 'acis_oi_spectrogram.png'
plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
plt.close()
fig.clf()
plt.close(fig)
plt.close('all')