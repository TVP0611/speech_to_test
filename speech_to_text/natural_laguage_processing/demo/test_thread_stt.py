
import os
from playsound import playsound
import speech_recognition as sr
import time
import sys
import ctypes
import wikipedia
import datetime
import json
import re
import webbrowser
# import smtplib
import requests
import urllib
import urllib.request as urllib2
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from webdriver_manager.chrome import ChromeDriverManager
from time import strftime
from gtts import gTTS
from youtube_search import YoutubeSearch
import vlc
import pafy
import threading
import time
from io import BytesIO
import numpy as np
import base64
import sounddevice as sd
import soundfile as sf
# from pygame import mixer
from pydub import AudioSegment
from pydub.playback import play


room_master = 'Phúc'
name_butler = 'mimi'
wikipedia.set_lang('vi')
language = 'vi'
path = ChromeDriverManager().install()
stop_event = threading.Event()

def speak_feedback(text):
    print("Bot: {}".format(text))
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("sound1.mp3")
    player = vlc.MediaPlayer("sound1.mp3")
    # player.play()
    # good_states = ["State.Playing", "State.NothingSpecial", "State.Opening"]
    # while str(player.get_state()) in good_states:
    #     # time.sleep(6)
    #     # print("done")
    #     # pass
    playsound("sound1.mp3", False)
    time.sleep(3)
    # player.stop()
    os.remove("sound1.mp3")

def speak_thread(text, stop_event):
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("sound.mp3")
    player = vlc.MediaPlayer("sound.mp3")
    player.play()
    # time.sleep(2)
    good_states = ["State.Playing", "State.NothingSpecial", "State.Opening"]
    while str(player.get_state()) in good_states:
        if stop_event.is_set():
            player.stop()
            os.remove("sound.mp3")
            stop_event.clear()
            sys.exit()
    player.stop()
    os.remove("sound.mp3")

def get_audio():
    while True:
        r = sr.Recognizer()
        with sr.Microphone() as source:
            print("Tôi: ", end='')
            audio = r.listen(source, phrase_time_limit=5)
            try:
                text = r.recognize_google(audio, language="vi-VN")
                print(text)
                if text == name_butler:
                    stop_event.set()
                return text
            except:
                print("...")
                return 0

def hello(name):
    day_time = int(strftime('%H'))
    if day_time < 12:
        speak_feedback("Chào buổi sáng bạn {}. Chúc bạn một ngày tốt lành.".format(name))
    elif 12 <= day_time < 18:
        speak_feedback("Chào buổi chiều bạn {}. Bạn đã dự định gì cho chiều nay chưa.".format(name))
    else:
        speak_feedback("Chào buổi tối bạn {}. Bạn đã ăn tối chưa nhỉ.".format(name))


def stop():
    speak_feedback("Hẹn gặp lại bạn sau!")

def get_text():
    while True:
        text = get_audio()
        if text:
            return text.lower()
        else:
            # speak_feedback("Mimi không nghe rõ. Bạn nói lại được không!")
            time.sleep(1)
            pass

def play_song(stop_event):
    speak_feedback('Xin mời bạn chọn tên bài hát')
    mysong = get_text()
    while True:
        result = YoutubeSearch(mysong, max_results=5).to_dict()
        if result:
            break
    ############# sẽ có if result = None trả lời tôi không kiếm đc
    url = 'https://www.youtube.com' + result[0]['url_suffix']
    # webbrowser.open(url)
    speak_feedback("Bài hát bạn yêu cầu đã được mở.")
    video = pafy.new(url)
    best = video.getbestaudio()
    playurl = best.url
    ins = vlc.Instance()
    player = ins.media_player_new()
    code = urllib.request.urlopen(url).getcode()
    if str(code).startswith('2') or str(code).startswith('3'):
        print('Stream is working')
    else:
        print('Stream is dead')
    Media = ins.media_new(playurl)
    Media.get_mrl()
    player.set_media(Media)
    player.play()
    good_states = ["State.Playing", "State.NothingSpecial", "State.Opening"]
    while str(player.get_state()) in good_states:
        # print('Stream is working. Current state = {}'.format(player.get_state()))
        if stop_event.is_set():
            # if os.path.isfile('sound.mp3'):
            player.stop()
            # os.remove("sound.mp3")
            stop_event.clear()
            # player.stop()
            sys.exit()

    # print('Stream is not working. Current state = {}'.format(player.get_state()))
    player.stop()
    
def current_weather(stop_event):
    speak_feedback("Bạn muốn xem thời tiết ở đâu ạ.")
    ow_url = "http://api.openweathermap.org/data/2.5/weather?"
    city =  get_text() #input("thành phố")
    if not city:
        pass
    api_key = "fe8d8c65cf345889139d8e545f57819a"
    call_url = ow_url + "appid=" + api_key + "&q=" + city + "&units=metric"
    response = requests.get(call_url)
    data = response.json()
    if data["cod"] != "404":
        city_res = data["main"]
        current_temperature = city_res["temp"]
        current_pressure = city_res["pressure"]
        current_humidity = city_res["humidity"]
        suntime = data["sys"]
        sunrise = datetime.datetime.fromtimestamp(suntime["sunrise"])
        sunset = datetime.datetime.fromtimestamp(suntime["sunset"])
        wthr = data["weather"]
        weather_description = wthr[0]["description"]
        now = datetime.datetime.now()
        content = """
        Hôm nay là ngày {day} tháng {month} năm {year}
        Mặt trời mọc vào {hourrise} giờ {minrise} phút
        Mặt trời lặn vào {hourset} giờ {minset} phút
        Nhiệt độ trung bình là {temp} độ C
        Áp suất không khí là {pressure} héc tơ Pascal
        Độ ẩm là {humidity}%
        Trời hôm nay quang mây. Dự báo mưa rải rác ở một số nơi.""".format(day = now.day,month = now.month, year= now.year, hourrise = sunrise.hour, minrise = sunrise.minute,                                                              hourset = sunset.hour, minset = sunset.minute, 
                                                                           temp = current_temperature, pressure = current_pressure, humidity = current_humidity)
        # speak_thread(content, None)
        # time.sleep(20)
        tts = gTTS(text=content, lang=language, slow=False)
        tts.save("sound.mp3")
        player = vlc.MediaPlayer("sound.mp3")
        player.play()
        time.sleep(2)
        good_states = ["State.Playing", "State.NothingSpecial", "State.Opening"]
        while str(player.get_state()) in good_states:
            if stop_event.is_set():
                player.stop()
                os.remove("sound.mp3")
                stop_event.clear()
                sys.exit()
        player.stop()
        os.remove("sound.mp3")
    else:
        speak_feedback("Không tìm thấy địa chỉ của bạn")
    # if stop_event.is_set():
    #     print("dừng thời tiết.")
    #     sys.exit()



def assistant():
    action_sentence = get_text()
    if "thời tiết" in action_sentence:
        threading.Thread(name="weather", target= current_weather, args=(stop_event, ), daemon= True).start()
    elif "chào" in action_sentence:
        hello(room_master)
    elif "mở nhạc" in action_sentence:
        threading.Thread(name="music", target= play_song, args=(stop_event, ), daemon= True).start()
    else: pass

def wake_word():
    text =  get_text()  #input(">> ")     get_text()
    if name_butler in text:
        for t in threading.enumerate():
            # print(t)
            if t.getName() == "weather":
                stop_event.set()
            elif t.getName() == "music":
                stop_event.set()
        speak_feedback("dạ vâng có em đây")
        assistant()
    else:
        pass

while True:
    wake_word()