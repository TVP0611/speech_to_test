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

# input = "Một số tổ chức y tế tại Na Uy tuyên bố nước này thoát khỏi Covid-19. Nhiều người Việt tại đây lại chia sẻ cuộc sống ít bị ảnh hưởng bởi đại dịch, thay vào đó là nỗi lo trầm cảm."
#
# tts = gTTS(text=input, lang="vi", slow=False)
# tts.save("sound.mp3")
# player = vlc.MediaPlayer("D:/train_model_speech_to_test/speech_to_text/process_voice_classify/test_audio/bat_den_ap_tran.wav")
# player.play
# # time.sleep(2)
# good_states = ["State.Playing", "State.NothingSpecial", "State.Opening"]
# while str(player.get_state()) in good_states:
#     print(player.get_state())
#     # if event.is_set():
#     #     player.stop()
#     #     os.remove("sound.mp3")
#     #     event.clear()
#     #     sys.exit()
# player.stop()


from pydub import AudioSegment
from pydub.playback import play

#Load an audio file
myAudioFile = 'test_invert.wav'
sound1 = AudioSegment.from_file(myAudioFile, format="wav")

sound3 = AudioSegment.from_file('speech_test_stt_gg.wav', format="wav")

#Invert phase of audio file
sound2 = sound1.invert_phase()

#Merge two audio files
combined = sound3.overlay(sound2)

#Export merged audio file
combined.export("outAudio.wav", format="wav")

#Play audio file :
#should play nothing since two files with inverse phase cancel each other
mergedAudio = AudioSegment.from_wav("outAudio.wav")
play(mergedAudio)
