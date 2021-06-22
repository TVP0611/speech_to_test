# with open('D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/bat_den_ap_tran.flac', 'rb') as f:
#     flac_data = f.read()
#
#
# with open('D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/bat_den_ap_tran.wav', 'rb') as n:
#     wav_data = n.read()
#
# print("done")

import pyaudio
import numpy as np
import wave
import io
import subprocess
import audioop
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

def append_silence(data_new):
    for x in range(int(3000)):
        data_new.insert(0, 0)
    if len(data_new) < 8000:
        for y in range(int(8000-len(data_new))):
            data_new.append(0)
    return data_new

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
    if not isinstance(actual_result, dict) or len(actual_result.get("alternative", [])) == 0: raise UnknownValueError()

    if "confidence" in actual_result["alternative"]:
        # return alternative with highest confidence score
        best_hypothesis = max(actual_result["alternative"], key=lambda alternative: alternative["confidence"])
    else:
        # when there is no confidence available, we arbitrarily choose the first hypothesis.
        best_hypothesis = actual_result["alternative"][0]
    if "transcript" not in best_hypothesis: raise UnknownValueError()
    text = best_hypothesis["transcript"]
    return text

p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=8000,
                input=True, frames_per_buffer=1024)
wait = 0

data_array = []
while True:
    #####      Bắt tín hiệu audio
    data1 = stream.read(1024)
    data = np.frombuffer(data1, dtype=np.int16)
    data_int = bytes(data)
    # print(max(data))
    #####     Nhận biết tiếng nói và Cắt câu
    # if max(data) >= 1000:
        # data = normalize(data)
        # data = filter_audio(sr, data)
        # data = data.astype(np.int16)
    energy = audioop.rms(data1, 2)
    data_array.extend(data1)
    if len(data_array) > 7000:
        wait = 0
        with io.BytesIO() as wav_file:
            wav_writer = wave.open(wav_file, "wb")
            try:  # note that we can't use context manager, since that was only added in Python 3.4
                wav_writer.setframerate(8000)
                wav_writer.setsampwidth(2)
                wav_writer.setnchannels(1)
                wav_writer.writeframes(data1)
                wav_data = wav_file.getvalue()
                print("wav: ", wav_data)
                flac_converter = 'C:/anaconda3/envs/speech_to_text/lib/site-packages/speech_recognition/flac-win32.exe'
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
                # print("flac: ", flac_data)
                # text = convert_stt(flac_data)
                # print(text)
            finally:  # make sure resources are cleaned up
                wav_writer.close()
        data_array.clear()


