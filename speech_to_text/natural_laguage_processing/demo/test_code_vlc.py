import vlc
from gtts import gTTS
import time
import os
import threading
import sys
from playsound import playsound
import speech_recognition as sr
from multiprocessing import Process

name_butler = 'mimi'
# wikipedia.set_lang('vi')
language = 'vi'
stop_event = threading.Event()
stop = False
pause_event = threading.Event()
words = []


def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Tôi: ", end='')
        audio = r.listen(source, phrase_time_limit=8)
        try:
            text = r.recognize_google(audio, language="vi-VN")
            print(text)
            return text
        except:
            print("...")
            return 0

def get_text():
    while True:
        text = get_audio()
        if text:
            return text.lower()
        else:
            speak_feedback("Mimi không nghe rõ. Bạn nói lại được không!")
            time.sleep(1)
            return "0"

def speak_feedback(text):
    print("Bot: {}".format(text))
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save("sound1.mp3")
    playsound("sound1.mp3", False)
    time.sleep(2)
    os.remove("sound1.mp3")

# def speak(text, stop_event):
#     flag =  stop_event.is_set()
#     print("Bot: {}".format(text))
#     tts = gTTS(text=text, lang=language, slow=False)
#     tts.save("sound.mp3")
#     # playsound("sound.mp3", False)
#     player = vlc.MediaPlayer("sound.mp3")
#     if flag == True:
#         # if os.path.isfile('sound.mp3'):
#         player.stop()
#         # os.remove("sound.mp3")
#         stop_event.clear()
#         # player.stop()
#         sys.exit()
#     player.play()
#     good_states = ["State.Playing", "State.NothingSpecial", "State.Opening"]
#     while str(player.get_state()) in good_states:
#         print(str(player.get_state()))
#         # pass
#     # time.sleep(2)
#     player.stop()
#     # if os.path.isfile('sound.mp3'):
#     #     player.stop()
#     #     time.sleep(1)
#     #     os.remove("sound.mp3")
    

def story(stop_event):
    global stop, player
    # stop = stop_event.is_set()
    # pause = pause_event.is_set()
    # flag = stop_event.is_set
    sentence = """Quê hương em là một ngôi làng nhỏ phía bên bờ dòng sông Hồng.
                Nơi đây có cánh đồng lúa rộng mỏi cánh cò bay, có những vườn cây trĩu quả ngọt, 
                có những luống rau xanh mướt mắt… Chiều chiều, bên bờ đê, lại bay lên những cánh diều đủ hình thù màu sắc của lũ trẻ. 
                ại thấp thoáng những hơi khói mỏng manh bay lên từ căn bếp nhỏ. 
                Lại văng vẳng tiếng cười, tiếng nói của những gia đình nhỏ mà ấm áp.
                Ôi! Sao mà bình yên đến thế!"""
    # print(sentence)
    tts = gTTS(text=sentence, lang=language, slow=False)
    tts.save("sound.mp3")
    # playsound("sound.mp3", False)
    # time.sleep(5)
    # os.remove("sound.mp3")

    player = vlc.MediaPlayer("sound.mp3")
    player.play()
    time.sleep(2)
    good_states = ["State.Playing", "State.NothingSpecial", "State.Opening"]
    while str(player.get_state()) in good_states:
        # print(str(stop_event.is_set()))
        if stop_event.is_set():
            # if os.path.isfile('sound.mp3'):
            player.stop()
            # os.remove("sound.mp3")
            stop_event.clear()
            # player.stop()
            sys.exit()
            
    player.stop()
   
def assistant():
    action = input('action: ')
    if "kể chuyện" in action:
        t_story.start()
        # t_story = Process(target=story)    
        # t_story.start()
    else: pass

def wake_word():
    # global stop, player
    while True:
        
        text =  input(">> ")  #input(">> ")     get_text()
        if name_butler in text:
             
            # t = threading.enumerate()#### check lại
            for t in threading.enumerate():
                if t.getName() == "story":
                    # stop = True
                    # player.vlm_stop_media()
                    stop_event.set()
                    # flag = stop_event.is_set
                    # if flag == True:
                        # if os.path.isfile('sound.mp3'):
                        # player.stop()
                        # os.remove("sound.mp3")
                        # stop_event.clear()
                        # player.stop()
                        # sys.exit()
            speak_feedback("dạ vâng có em đây")
            assistant()
            # action = input('action: ')     #input('action: ')   get_text()
            # if "kể chuyện" in action:
            #     t_story.start()
            # else:
            #     pass
        else:
            pass
        
   
# wake_up = threading.Thread(name='wake', target=wake_word)
# wake_up.start()
while True:
    t_story = threading.Thread(name="story", target=story, args=(stop_event, ), daemon=True)
    wake_word()
    #  while True:
    # text =  input(">> ")  #input(">> ")     get_text()
    # if name_butler in text:
    #     t_story = threading.Thread(name="story", target=story, args=(stop_event, ), daemon=True) 
    #     # t = threading.enumerate()#### check lại
    #     for t in threading.enumerate():
    #         if t.getName() == "story":
    #             # stop = True
    #             # player.vlm_stop_media()
    #             stop_event.set()
    #             # flag = stop_event.is_set
    #             # if flag == True:
    #                 # if os.path.isfile('sound.mp3'):
    #                 # player.stop()
    #                 # os.remove("sound.mp3")
    #                 # stop_event.clear()
    #                 # player.stop()
    #                 # sys.exit()
    #     speak_feedback("dạ vâng có em đây")
    #     # assistant()
    #     action = input('action: ')     #input('action: ')   get_text()
    #     if "kể chuyện" in action:
    #         t_story.start()
    #     else:
    #         pass
    # else:
    #     pass