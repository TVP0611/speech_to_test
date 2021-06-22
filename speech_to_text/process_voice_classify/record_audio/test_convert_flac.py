# with open('D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/bat_den_ap_tran.flac', 'rb') as f:
#     flac_data = f.read()
#
#
# with open('D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/bat_den_ap_tran.wav', 'rb') as n:
#     wav_data = n.read()
#
# print("done")
from gtts import gTTS
import vlc, time

input = "Một số tổ chức y tế tại Na Uy tuyên bố nước này thoát khỏi Covid-19. Nhiều người Việt tại đây lại chia sẻ cuộc sống ít bị ảnh hưởng bởi đại dịch, thay vào đó là nỗi lo trầm cảm."

tts = gTTS(text=input, lang="vi", slow=False)
tts.save("sound.mp3")
player = vlc.MediaPlayer("D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/bat_den_ap_tran.wav")
player.play
# time.sleep(2)
good_states = ["State.Playing", "State.NothingSpecial", "State.Opening"]
while str(player.get_state()) in good_states:
    print(player.get_state())
    # if event.is_set():
    #     player.stop()
    #     os.remove("sound.mp3")
    #     event.clear()
    #     sys.exit()
player.stop()
