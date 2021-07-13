import sys
import os
import timeit
import os.path
sys.path.append(os.path.abspath('./hotword_detection'))
from hotword_detection import wordRecorder as wr
from hotword_detection import hwDetector as hd
from scipy.io import wavfile

array_data = []
wrdRec = wr.wordRecorder()
hwDet = hd.hwDetector()
file = os.listdir("New folder")
for i in file:
    _fs, _data = wavfile.read('./New folder/' + i)
    array_data.append(_data)
print("Start")

for d in array_data:
    # print("Speak a word")
    # wrdRec.record2File("demo.wav")
    start = timeit.default_timer()
    print(hwDet.isHotword_in_ram(d))
    stop = timeit.default_timer()
    print(stop - start, " (seconds)")
