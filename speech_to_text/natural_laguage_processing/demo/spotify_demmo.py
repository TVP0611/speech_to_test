import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
import webbrowser
import vlc
import urllib.request
import pafy
import vlc
# import pafy
import urllib.request
import threading
import time
import sys

def music(stop_event, event_obj ):
    url = "https://www.youtube.com/watch?v=ZXddoGmHXw8"
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
        return

    Media = ins.media_new(playurl)
    Media.get_mrl()
    player.set_media(Media)
    player.play()
    good_states = ["State.Playing", "State.NothingSpecial", "State.Opening"]
    while str(player.get_state()) in good_states:
        player.get_state()
        if event_obj.is_set():
            player.pause()
            print('thread await')
            time.sleep(5)
            print('end wait')
            event_obj.clear()
            player.play()
            
        elif stop_event.is_set():
            print ("Thread has been interrupted by an event.")
            player.stop()
            sys.exit()
            # return
        # print ("Executing.")

    # good_states = ["State.Playing", "State.NothingSpecial", "State.Opening"]
    # while str(player.get_state()) in good_states:
    #     print('Stream is working. Current state = {}'.format(player.get_state()))

    # print('Stream is not working. Current state = {}'.format(player.get_state()))
    player.stop()

event_obj = threading.Event()
stop_event = threading.Event()

t = threading.Thread(target=music, args=(stop_event, event_obj, ))
t.start()
i = 0
while 1:
    i += 1
    sentences = input('>>')
    if sentences == 'stop':
        # time.sleep(2.0)
        stop_event.set()
        # break
    elif sentences == 'pause':
        event_obj.set()


# spotify = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id="bb5d1b57fc27426bb92ca98cb9d86f78",
#                                                             client_secret="da767dbb0be24a7d9453dde1351037bd"))
# # devices = spotify.devices()
# deviceID = devices['devices'][0]['id']
# results = spotify.search(q='my_tam', limit=5)
# for idx, track in enumerate(results['tracks']['items']):
#     m = (idx, track['name'])
# spotify.start_playback(None, 0, 0)
# print(track['name'])
# webbrowser.register('chrome', None)
# webbrowser.open_new('https://open.spotify.com/track/3CNGmZVbvs4UVY2fWHXBvD')
# p = vlc.MediaPlayer("https://open.spotify.com/track/3CNGmZVbvs4UVY2fWHXBvD")
# p.play()
# webUrl  = urllib.request.urlopen("https://open.spotify.com/track/3CNGmZVbvs4UVY2fWHXBvD")
# print()
