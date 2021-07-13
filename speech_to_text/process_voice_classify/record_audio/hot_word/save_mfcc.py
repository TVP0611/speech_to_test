from python_speech_features import mfcc
from os import listdir
from os.path import isdir, join
import librosa
import random
import numpy as np
import matplotlib.pyplot as plt

# target_list = all_targets
feature_sets_file = 'acis_mfcc.npz'
perc_keep_samples = 1.0 # 1.0 is keep all samples
val_ratio = 0.1
test_ratio = 0.1
sample_rate = 8000
num_mfcc = 16
len_mfcc = 16

def extract_features(y, sr=8000, nfilt=13, winsteps=0.02):
    try:
        feat = mfcc(y, sr, nfilt=nfilt, winstep=winsteps)
        return feat
    except:
        raise Exception("Extraction feature error")

# def extract_features(in_files, in_y):
#     prob_cnt = 0
#     out_x = []
#     out_y = []
#
#     for index, filename in enumerate(in_files):
#
#         # Create path from given filename and target item
#         path = join(dataset_path, target_list[int(in_y[index])],
#                     filename)
#
#         # Check to make sure we're reading a .wav file
#         if not path.endswith('.wav'):
#             continue
#
#         # Create MFCCs
#         mfccs = calc_mfcc(path)
#
#         # Only keep MFCCs with given length
#         if mfccs.shape[1] == len_mfcc:
#             out_x.append(mfccs)
#             out_y.append(in_y[index])

def calc_mfcc(path):
    # Load wavefile
    signal, fs = librosa.load(path, sr=sample_rate)

    # Create MFCCs from sound clip
    mfccs = extract_features(signal)
    mfccs = np.resize(mfccs, (32, 13))
    return mfccs#.transpose()


if __name__ == '__main__':
    trainDir = './train_audio/'
    mfcc_acis = []
    label_acis = []
    for file_name in listdir(trainDir):
        if file_name.endswith(".wav"):
            _mfccs = calc_mfcc(trainDir + file_name)
            mfcc_acis.append(_mfccs)
            label_acis.append(0)
    np.savez(feature_sets_file, x=mfcc_acis, y=label_acis)
    feature_sets = np.load(feature_sets_file)
    feature_sets.files
    f = feature_sets['x']
    print('done')


