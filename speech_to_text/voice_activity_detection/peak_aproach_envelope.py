# from scipy.io import wavfile
# from scipy.signal import hilbert
# import matplotlib.pyplot as plt
# import numpy as np
#
# sr, signal = wavfile.read("audio/mdpk_audio.wav")
#
# # duration = int(len(signal)/sr)
# # samples = int(sr * duration)
# t = np.arange(len(signal)) / sr
#
# analytic_signal = hilbert(signal)
# amplitude_envelope = np.around(np.abs(analytic_signal), decimals=1)
#
# plt.plot(t, signal, label='signal')
# plt.plot(t, amplitude_envelope, label='envelope')
# plot1 = plt.figure()


import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

audio_path = "audio/audio_450ms.wav"#"D:/train_model_speech_to_test/speech_to_text/project_model_recognition_images_spectrogram/data_augment/bat/bat_8_up_vol_9.wav"#"audio/audio_450ms.wav"

data, sample_rate = librosa.load(path=audio_path, sr=8000)
fig = plt.figure(figsize=[0.72,0.72])
ax = fig.add_subplot(111)
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.set_frame_on(False)
S = librosa.feature.melspectrogram(y=data, sr=sample_rate, n_fft=512, hop_length=128)
Spec = librosa.amplitude_to_db(S, ref=np.max)
# Spec_normalize = np.array(Spec, dtype="float") / (255.0)
# Averge_col = Spec_normalize.mean(axis=0)
# averge = sum(Averge_col)/len(Averge_col)
librosa.display.specshow(Spec)
#
# plot2 = plt.figure()
# plt.show()

plt.savefig("spec_audio_450ms.png", dpi=400, bbox_inches='tight', pad_inches=0)
# close thread of matplotlib (affect memory of pc)
plt.close()
fig.clf()
plt.close(fig)
plt.close('all')

print("done")