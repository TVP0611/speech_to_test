import pyaudio
from scipy.io import wavfile
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
from scipy.signal import butter, lfilter
from itertools import chain
from filter_noise import append_silence, filter_audio
import threading
# from code_filter_and_normalize import butter_bandpass_filter, append_silence
from dvg_ringbuffer import RingBuffer
# from test_model import *
import wave
import io
import subprocess
import time
import audioop
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

def convert_stt(flac_data):
    class RequestError(Exception): pass
    class UnknownValueError(Exception): pass

    language="vi-VN"

    key = "AIzaSyBOti4mM-6x9WDnZIjIeyEU21OpBXqWBgw"
    url = "http://www.google.com/speech-api/v2/recognize?{}".format(urlencode({
        "client": "chromium",
        "lang": language,
        "key": key,
    }))
    request = Request(url, data=flac_data, headers={"Content-Type": "audio/x-flac; rate={}".format(8000)})

    # obtain audio transcription results
    try:
        response = urlopen(request, timeout=None)
    except HTTPError as e:
        raise RequestError("recognition request failed: {}".format(e.reason))
    except URLError as e:
        raise RequestError("recognition connection failed: {}".format(e.reason))
    response_text = response.read().decode("utf-8")

    # ignore any blank blocks
    actual_result = []
    for line in response_text.split("\n"):
        if not line: continue
        result = json.loads(line)["result"]
        if len(result) != 0:
            actual_result = result[0]
            break

    # return results
    if not isinstance(actual_result, dict) or len(actual_result.get("alternative", [])) == 0:
        #raise UnknownValueError()
        return None

    if "confidence" in actual_result["alternative"]:
        # return alternative with highest confidence score
        best_hypothesis = max(actual_result["alternative"], key=lambda alternative: alternative["confidence"])
    else:
        # when there is no confidence available, we arbitrarily choose the first hypothesis.
        best_hypothesis = actual_result["alternative"][0]
    if "transcript" not in best_hypothesis: raise UnknownValueError()
    text = best_hypothesis["transcript"]
    return text

def convert_array_audio_to_flac(data):
    with io.BytesIO() as wav_file:
        wav_writer = wave.open(wav_file, "wb")
        try:  # note that we can't use context manager, since that was only added in Python 3.4
            wav_writer.setframerate(8000)
            wav_writer.setsampwidth(2)
            wav_writer.setnchannels(1)
            wav_writer.writeframes(data)
            wav_data = wav_file.getvalue()
            # print("wav: ", wav_data)
            flac_converter = "D:/train_model_speech_to_test/speech_to_text/process_voice_classify/record_audio/convert_flac/flac.exe"    #'C:/anaconda3/envs/speech_to_text/lib/site-packages/speech_recognition/flac-win32.exe'
            startup_info = subprocess.STARTUPINFO()
            startup_info.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # specify that the wShowWindow field of `startup_info` contains a value
            startup_info.wShowWindow = subprocess.SW_HIDE
            process = subprocess.Popen([
                flac_converter,
                "--stdout", "--totally-silent",
                # put the resulting FLAC file in stdout, and make sure it's not mixed with any program output
                "--best",  # highest level of compression available
                "-",  # the input FLAC file contents will be given in stdin
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE, startupinfo=startup_info)
            flac_data, stderr = process.communicate(wav_data)
            wav_writer.close()
            return flac_data
        except:
            pass

# kss = Keyword_Spotting_Service()
# kss1 = Keyword_Spotting_Service()

# check that different instances of the keyword spotting service point back to the same object (singleton)
# assert kss is kss1

######      Khởi tạo giá trị khi record (tỉ lệ lấy mẫu = 8000 hz, channel = 1, giá trị tín hiệu theo kiểu = int16)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=8000,
                input=True, frames_per_buffer=1024)

rb = RingBuffer(640000, dtype=np.int16)
pt = 0
val_wait = 0
signals = []
low = 100.0
high = 3900.0
sr = 8000
file_save = 0
audio = []
audio_split = []
data_new = []
data_flac = []
# count_pulse = []

def record():
    global pt, val_wait, audio
    while True:
        #####      Bắt tín hiệu audio
        audio_bytes = stream.read(1024)
        data = np.frombuffer(audio_bytes, dtype=np.int16)
        # print(max(data))
        ######     Nhận biết tiếng nói và Cắt câu
        if max(data) >= 1000:
            # data = normalize(data)
            # data = filter_audio(sr, data)
            data = data.astype(np.int16)
            # ls.extend(data)
            # rb.extend(data)
            rb.extend(data)
            val_wait = 0

        else:
            val_wait += 1
            # print(val_wait)
            if val_wait == 60:
                rb.clear()
                pt = 0

t2 = threading.Thread(target=record, daemon=True)
######      Bắt đầu chạy thread
t2.start()

while 1:
    if len(np.array(rb)) >= 1024:
        signals = rb[pt: pt + 64]
        if len(signals) >= 64:
            pt = pt + 64
            thresh_val = max(signals)
            if thresh_val >= 700:
                data_new.append(signals)
            elif thresh_val < 700:  # and a == 0:
                if len(data_new) > 5:
                    # print(len(data_new))
                    data_new = list(chain.from_iterable(data_new))
                    data_new = append_silence(data_new)
                    data_new1 = np.array(data_new)#.astype(np.int16)
                    # keyword, acc = kss.predict(data_new1)  # test_audio/bat/bat_42.wav
                    # print(keyword, max(acc) * 100)
                    # data_new = np.array(data_new).astype(np.int16)
                    ''''''
                    data_flac.extend(data_new)

                    data_flac = np.array(data_flac).astype(np.int16)
                    data_bytes = bytes(data_flac)
                    flac_converter = convert_array_audio_to_flac(data_bytes)
                    text = convert_stt(flac_converter)
                    print(text)
                    data_flac = []
                    # data_new = np.array(data_new).astype(np.int16)
                    # sf.write("D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/test_realtime/file_audio_rt_{0}.wav".format(file_save), data_new, sr, 'PCM_16')
                    data_new = []
                    file_save += 1
                    print('save done {}'.format(file_save))