import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import pickle

model_path = "ModelTFLite_DynamicRangeQuantization.tflite"

# Load model (interpreter)
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
# print(input_details)
# print(output_details)

# Load the image
image = cv2.imread("D:/train_model_speech_to_test/speech_to_text/project_model_recognition_images_spectrogram/data_spectrograms/bat/bat_0_down_vol_4.png")
output = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
# image = image.astype("int8")
image = np.expand_dims(image, axis=0)

lb = pickle.loads(open("lb.pickle", "rb").read())

interpreter.set_tensor(input_details[0]['index'], image)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

idx = np.argmax(output_data)
label = lb.classes_[idx]
acc = round(output_data[0][idx], 2)*100
print("name: {0}".format(label), ";", "acc: {0}".format(acc))





