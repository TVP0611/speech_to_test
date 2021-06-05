# from os import listdir
# from os.path import isdir, join
# import librosa
# import random
# import numpy as np
# import matplotlib.pyplot as plt
# import python_speech_features

# dataset_path = 'D:/Project/process_voice/process_voice_classify/audio_sample'
# # for name in listdir(dataset_path):
# #     if isdir(join(dataset_path, name)):
# #         # print(name)

# # Create an all targets list
# all_targets = [name for name in listdir(dataset_path) if isdir(join(dataset_path, name))]
# # print(all_targets)

# # See how many files are in each
# num_samples = 0
# for target in all_targets:
#     # print(len(listdir(join(dataset_path, target))))
#     num_samples += len(listdir(join(dataset_path, target)))
# print('Total samples:', num_samples)

# # Settings
# target_list = all_targets
# feature_sets_file = 'all_targets_mfcc_sets.npz'
# perc_keep_samples = 1.0 # 1.0 is keep all samples
# val_ratio = 0.1
# test_ratio = 0.1
# sample_rate = 8000
# num_mfcc = 16
# len_mfcc = 16

# # Create list of filenames along with ground truth vector (y)
# filenames = []
# y = []
# for index, target in enumerate(target_list):
#     # print(join(dataset_path, target))
#     filenames.append(listdir(join(dataset_path, target)))
#     y.append(np.ones(len(filenames[index])) * index)

# # Check ground truth Y vector
# # print(y)
# # for item in y:
# #     print(len(item))

# # Flatten filename and y vectors
# filenames = [item for sublist in filenames for item in sublist]
# y = [item for sublist in y for item in sublist]

# # Associate filenames with true output and shuffle
# filenames_y = list(zip(filenames, y))
# random.shuffle(filenames_y)
# filenames, y = zip(*filenames_y)

# # Only keep the specified number of samples (shorter extraction/training)
# print(len(filenames))
# filenames = filenames[:int(len(filenames) * perc_keep_samples)]
# print(len(filenames))

# # Calculate validation and test set sizes
# val_set_size = int(len(filenames) * val_ratio)
# test_set_size = int(len(filenames) * test_ratio)

# # Break dataset apart into train, validation, and test sets
# filenames_val = filenames[:val_set_size]
# filenames_test = filenames[val_set_size:(val_set_size + test_set_size)]
# filenames_train = filenames[(val_set_size + test_set_size):]

# # Break y apart into train, validation, and test sets
# y_orig_val = y[:val_set_size]
# y_orig_test = y[val_set_size:(val_set_size + test_set_size)]
# y_orig_train = y[(val_set_size + test_set_size):]

# # Function: Create MFCC from given path
# def calc_mfcc(path):
    
#     # Load wavefile
#     signal, fs = librosa.load(path, sr=sample_rate)
    
#     # Create MFCCs from sound clip
#     mfccs = python_speech_features.base.mfcc(signal, 
#                                             samplerate=fs,
#                                             winlen=0.256,
#                                             winstep=0.050,
#                                             numcep=num_mfcc,
#                                             nfilt=26,
#                                             nfft=2048,
#                                             preemph=0.0,
#                                             ceplifter=0,
#                                             appendEnergy=False,
#                                             winfunc=np.hanning)
#     return mfccs.transpose()

# # TEST: Construct test set by computing MFCC of each WAV file
# prob_cnt = 0
# x_test = []
# y_test = []
# for index, filename in enumerate(filenames_train):
    
#     # Stop after 500
#     if index >= 500:
#         break
    
#     # Create path from given filename and target item
#     path = join(dataset_path, target_list[int(y_orig_train[index])], 
#                 filename)
    
#     # Create MFCCs
#     mfccs = calc_mfcc(path)
    
#     if mfccs.shape[1] == len_mfcc:
#         x_test.append(mfccs)
#         y_test.append(y_orig_train[index])
#     else:
#         print('Dropped:', index, mfccs.shape)
#         prob_cnt += 1

# print('% of problematic samples:', prob_cnt / 500)

# # TEST: Test shorter MFCC
# # !pip install playsound
# from playsound import playsound

# idx = 13

# # Create path from given filename and target item
# path = join(dataset_path, target_list[int(y_orig_train[idx])], 
#             filenames_train[idx])

# # Create MFCCs
# mfccs = calc_mfcc(path)
# print("MFCCs:", mfccs)

# # Plot MFCC
# fig = plt.figure()
# plt.imshow(mfccs, cmap='inferno', origin='lower')

# # TEST: Play problem sounds
# print(target_list[int(y_orig_train[idx])])
# playsound(path)


# # Function: Create MFCCs, keeping only ones of desired length
# def extract_features(in_files, in_y):
#     prob_cnt = 0
#     out_x = []
#     out_y = []
        
#     for index, filename in enumerate(in_files):
    
#         # Create path from given filename and target item
#         path = join(dataset_path, target_list[int(in_y[index])], 
#                     filename)
        
#         # Check to make sure we're reading a .wav file
#         if not path.endswith('.wav'):
#             continue

#         # Create MFCCs
#         mfccs = calc_mfcc(path)

#         # Only keep MFCCs with given length
#         if mfccs.shape[1] == len_mfcc:
#             out_x.append(mfccs)
#             out_y.append(in_y[index])
#         else:
#             print('Dropped:', index, mfccs.shape)
#             prob_cnt += 1
            
#     return out_x, out_y, prob_cnt

# # Create train, validation, and test sets
# x_train, y_train, prob = extract_features(filenames_train, 
#                                           y_orig_train)
# print('Removed percentage:', prob / len(y_orig_train))
# x_val, y_val, prob = extract_features(filenames_val, y_orig_val)
# print('Removed percentage:', prob / len(y_orig_val))
# x_test, y_test, prob = extract_features(filenames_test, y_orig_test)
# print('Removed percentage:', prob / len(y_orig_test))

# # Save features and truth vector (y) sets to disk
# np.savez(feature_sets_file, 
#          x_train=x_train, 
#          y_train=y_train, 
#          x_val=x_val, 
#          y_val=y_val, 
#          x_test=x_test, 
#          y_test=y_test)

# # TEST: Load features
# feature_sets = np.load(feature_sets_file)
# feature_sets.files

# len(feature_sets['x_train'])

# print(feature_sets['y_val'])

import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image as im
import os
import python_speech_features

n_mels = 256
hop_length = 512

n_fft = 1024
path = 'D:/Project/process_voice/process_voice_classify/audio_data/'
path_save = 'D:/Project/process_voice/process_voice_classify/test_spec/'
file_name = 'bat'
num_mfcc = 16

entries = os.listdir(path)
for i in entries:
    # print(path + '{0}/*.wav'.format(i))
    if i == file_name:
        file = os.listdir(path + '{0}/'.format(i))
        cnt = 0
        for j in file:
            print(path + '{0}/{1}'.format(i, j))
            save_file = 0
            pt = 0
            y, sr = librosa.load(path + file_name + '/' + j, sr=8000)
            S = librosa.feature.melspectrogram(y, sr=int(sr), hop_length=hop_length, win_length=n_mels, n_fft=n_fft, power=2)
            S_DB = librosa.power_to_db(S, ref=np.max)
            # S_DB = S_DB.astype(np.int16)
            # h, w = S_DB.shape[0], S_DB.shape[1]
            # for x in range(h):
            #     for j in range(w):
            #         if S_DB[x][j] > -50:
            #             pass
            #         elif S_DB[x][j] <= -50:
            #             S_DB[x][j] = -80
            fig = plt.figure(figsize=[0.72, 0.72])
            ax = fig.add_subplot(111)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.set_frame_on(False)
            librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', fmax=5000, fmin=50)
            # plt.colorbar(format='%+2.0f dB')
            # plt.show()
            filename = path_save + file_name + '/' + '/{0}_{1}.png'.format(file_name, cnt)
            plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
            plt.close()
            cnt += 1
            # cv2.imshow('show', mfccs)
            # cv2.imwrite("test.png", mfccs)
            # cv2.waitKey(0)

print("done")

# Load sound file

# y, sr = librosa.load("file_mo_0/file_tat_0.wav", sr=8000)
# # y1, sr1 = librosa.load("audio_sample/tat/file_tat_14.wav", sr=8000)

# # D = np.abs(librosa.stft(y))
# # X = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, win_length=n_mels, window='hanning')
# # Y = np.abs(X)**4
# M = librosa.filters.mel(8000, n_fft=1024)
# S1 = np.abs(librosa.stft(y, hop_length=hop_length, win_length=n_mels, n_fft=n_fft))
# # S = librosa.feature.melspectrogram(S1, sr=sr, hop_length=hop_length, n_mels=n_mels, n_fft=n_fft, power=2)
# # S1 = librosa.feature.melspectrogram(y1, sr=sr1, hop_length=hop_length, n_mels=n_mels, n_fft=n_fft, power=2)

# # S = librosa.feature.melspectrogram(y, sr=sr, n_mels=n_mels, fmax=8000)
# # M = librosa.filters.mel(8000, n_fft=128)
# # S1 = S1.astype(np.int16)
# S_DB = librosa.amplitude_to_db(S1**2.5, ref=np.max)#np.max
# S_DB = S_DB.astype(np.int16)
# # S_DB1 = librosa.power_to_db(S1**2, ref=np.max)
# # S_DB1 = S_DB1.astype(np.int16)
# h, w = S_DB.shape[0], S_DB.shape[1]

# # S_DB[0][0] = 0
# # for i, x in enumerate(S_DB):
# #     aver = sum(x)/ len(x)
# #     # print(aver)
# #     for ind, e in enumerate(x):
# #         if e > aver/2.5:
# #             S_DB[i][ind] = 0
# #             # pass
# #         elif e <= aver/2.5:
# #             S_DB[i][ind] = -80


# # for x in range(h):
# #     for j in range(w):
# #         if S_DB[x][j] > -10:
# #             # if S_DB[x][j] > -28:
# #             #     pass
# #             # else:
# #             #     S_DB[x][j] = -80
# #             # S_DB[x][j] = 255           ##  S_DB[x][j]/2
# #             pass
# #         elif S_DB[x][j] <= -10:#or S_DB[x][j] == -28 or S_DB[x][j] == -35:
# #             S_DB[x][j] = -80

# # for k in range(h):
# #     for d in range(w):
# #         if S_DB1[k][d] > -60:
# #             # if S_DB[x][j] > -28:
# #             #     pass
# #             # else:
# #             #     S_DB[x][j] = -80
# #             # S_DB[x][j] = 255           ##  S_DB[x][j]/2
# #             pass
# #         elif S_DB1[k][d] <= -60:
# #             S_DB1[k][d] = -80



# # _, mask=cv2.threshold(gray, 125, 255, cv2.THRESH_BINARY_INV)
# # # Creating a 3x3 kernel
# # kernel = np.ones((3, 3), np.uint8)
# # # Performing dilation on the mask
# # dilation = cv2.dilate(S_DB, kernel)
# # # cv2.imshow('Spec', dilation)
# # # cv2.waitKey(0)
# # data = im.fromarray(dilation)
# # data.show()
# fig = plt.figure(figsize=[0.72, 0.72])
# ax = fig.add_subplot(111)
# ax.axes.get_xaxis().set_visible(False)
# ax.axes.get_yaxis().set_visible(False)
# ax.set_frame_on(False)
# librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel', fmax=5000, fmin=50)
# plt.colorbar(format='%+2.0f dB')
# plt.show()
# # filename = "file_mo_0/" + 'tat.png'
# # plt.savefig(filename, dpi=400, bbox_inches='tight', pad_inches=0)
# # librosa.display.specshow(S_DB1, sr=sr1, hop_length=hop_length, x_axis='time', y_axis='mel', fmax=5000, fmin=50)
# # plt.colorbar(format='%+2.0f dB')
# # plt.show()

# # mel = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
# #
# # plt.figure(figsize=(15, 4))
# #
# # plt.subplot(1, 3, 1)
# # librosa.display.specshow(mel, sr=sr, hop_length=hop_length, x_axis='linear')
# # plt.ylabel('Mel filter')
# # plt.colorbar()
# # plt.title('1. Our filter bank for converting from Hz to mels.')
# #
# # # plt.show()
# print("done")

