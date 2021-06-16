# import speech_recognition as sr
#
# r = sr.Recognizer()
# # while True:
#     # with sr.Microphone(sample_rate=8000) as source:
# with sr.AudioFile('D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/bat_den_ap_tran.wav') as source:
#     audio = r.listen(source, phrase_time_limit=5)
#     # print(audio.get_wav_data())
#     flac_data = audio.get_flac_data()
        # # with open('speech_test_stt_gg.wav', 'wb') as f:
        # #     f.write(audio.get_wav_data())
        # # print("done")

        # try:
        #     text = r.recognize_google(audio, language="vi-VN")
        #     print(text)
        #     # if text == name_butler:
        #     #     stop_event.set()
        #     # return text
        # except:
        #     print("...")
#             # return 0

import wave
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# wave.open('D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/bat_den_ap_tran.wav', 'rb')
# flac_data = wave.Wave_read.readframes()
# flac_data = flac_data[int((len(flac_data)-31951)):]

# import wave
#
#
# with wave.open('D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/bat_den_ap_tran.wav') as fd:
#     params = fd.getparams()
#     flac_data = fd.readframes(1000000) # 1 million frames max
#
# # print(params)

# w = wave.open('D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/bat_den_ap_tran.wav', 'rb')
# for i in range(w.getnframes()):
#     frame = w.readframes(i)
#
# print('done')

with open('D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/bat_den_ap_tran.flac', 'rb') as f:
    flac_data = f.read()
# print('done')
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


#!/usr/bin/env python3

# NOTE: this example requires PyAudio because it uses the Microphone class

# import time

# import speech_recognition as sr
#
#
# # this is called from the background thread
# def callback(recognizer, audio):
#     # received audio data, now we'll recognize it using Google Speech Recognition
#     try:
#         # for testing purposes, we're just using the default API key
#         # to use another API key, use `r.recognize_google(audio, key="GOOGLE_SPEECH_RECOGNITION_API_KEY")`
#         # instead of `r.recognize_google(audio)`
#         print("Google Speech Recognition thinks you said " + recognizer.recognize_google(audio, language="vi-VN"))
#     except sr.UnknownValueError:
#         print("Google Speech Recognition could not understand audio")
#     except sr.RequestError as e:
#         print("Could not request results from Google Speech Recognition service; {0}".format(e))
#
#
# r = sr.Recognizer()
# m = sr.Microphone()
# with m as source:
#     r.listen(source)  # we only need to calibrate once, before we start listening
#
# # start listening in the background (note that we don't have to do this inside a `with` statement)
# stop_listening = r.listen_in_background(m, callback)
# # `stop_listening` is now a function that, when called, stops background listening
#
# # do some unrelated computations for 5 seconds
# while 1:
#     # stop_listening = r.listen_in_background(m, callback)
#     pass  # we're still listening even though the main thread is doing other things
#
# # # calling this function requests that the background listener stop listening
# # stop_listening(wait_for_stop=False)
# #
# # # do some more unrelated things
# # while True: time.sleep(0.1)  # we're not listening anymore, even though the background thread might still be running for a second or two while cleaning up and stopping