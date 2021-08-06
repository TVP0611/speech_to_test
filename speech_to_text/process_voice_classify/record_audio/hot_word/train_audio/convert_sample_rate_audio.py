import librosa
import soundfile as sf
y, s = librosa.load('test_convert_audio_441khz_to_8khz.wav', sr=8000) # Downsample 44.1kHz to 8kHz
sf.write('audio_8khz.wav', y, 8000, 'PCM_16')
print('done')