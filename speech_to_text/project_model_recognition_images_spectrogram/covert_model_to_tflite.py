## Convert model to tflite
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import img_to_array
import random
from imutils import paths

# View website https://www.tensorflow.org/lite/performance/post_training_integer_quant#convert_to_a_tensorflow_lite_model

def convert_model2tflite_NoQuantization(model):
    converter = tf.lite.TFLiteConverter.from_saved_model(model, signature_keys=['serving_default'])
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.experimental_new_converter = True
    # converter.target_spec.supported_ops = [
    #     tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()

    fo = open("model.tflite", "wb")
    fo.write(tflite_model)
    fo.close

def convert_model2tflite_DynamicRangeQuantization(model):
    # Convert using dynamic range quantization
    converter = tf.lite.TFLiteConverter.from_saved_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model_DRQuant = converter.convert()

    tflite_model_quant_file = "ModelTFLite_DynamicRangeQuantization.tflite"
    fo = open(tflite_model_quant_file, "wb")
    fo.write(tflite_model_DRQuant)
    fo.close

def convert_model2tflite_Float16Quantization(model):
    # You can reduce the size of a floating point model by quantizing the weights to float16, the IEEE standard for 16-bit floating point numbers
    converter = tf.lite.TFLiteConverter.from_saved_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    tflite_model_Float16Quant = converter.convert()

    tflite_model_quant_file = "ModelTFLite_Float16Quantization.tflite"
    fo = open(tflite_model_quant_file, "wb")
    fo.write(tflite_model_Float16Quant)
    fo.close

def convert_model2tflite_UInt8Quantization(model, data_images):

    def representative_data_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(data_images).batch(1).take(100):
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_saved_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.int8 #change tf.uint8
    converter.inference_output_type = tf.int8 #change tf.uint8

    tflite_model_UInt8Quant = converter.convert()
    tflite_model_quant_file = "ModelTFLite_Int8Quantization.tflite" #
    fo = open(tflite_model_quant_file, "wb")
    fo.write(tflite_model_UInt8Quant)
    fo.close

def convert_model2tflite_Int16Quantization(model, data_images):

    def representative_data_gen():
        for input_value in tf.data.Dataset.from_tensor_slices(data_images).batch(1).take(100):
            yield [input_value]

    converter = tf.lite.TFLiteConverter.from_saved_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.int16
    converter.inference_output_type = tf.int16

    tflite_model_Int16Quant = converter.convert()
    tflite_model_quant_file = "ModelTFLite_Int16Quantization.tflite" #
    fo = open(tflite_model_quant_file, "wb")
    fo.write(tflite_model_Int16Quant)
    fo.close


if __name__ == '__main__':
    path_model = "D:/train_model_speech_to_test/speech_to_text/project_model_recognition_images_spectrogram/modelVGGnet_SpecRecog.model"
    path_images = "D:/train_model_speech_to_test/speech_to_text/project_model_recognition_images_spectrogram/data_spectrograms"

    print("[INFO] loading images...")
    # # initialize the data
    # IMAGE_DIMS = (96, 96, 3)
    # data = []
    #
    # # grab the image paths and randomly shuffle them
    # imagePaths = sorted(list(paths.list_images(path_images)))
    # random.seed(42)
    # random.shuffle(imagePaths)
    #
    # # loop over the input images
    # for imagePath in imagePaths:
    #     # load the image, pre-process it, and store it in the data list
    #     image = cv2.imread(imagePath)
    #     image = cv2.resize(image, (IMAGE_DIMS[1], IMAGE_DIMS[0]))
    #     image = img_to_array(image)
    #     data.append(image)
    #
    # # scale the raw pixel intensities to the range [0, 1]
    # data = np.array(data, dtype=np.float32) / 255.0

    ## save data
    # np.save('data_specs.npy', data)

    ## load data
    # data = np.load('data_specs.npy')
    print("[INFO] load data done.")

    # Convert model
    # convert_model2tflite_NoQuantization(model=path_model)
    # convert_model2tflite_DynamicRangeQuantization(model=path_model)
    # convert_model2tflite_Float16Quantization(model=path_model)
    # convert_model2tflite_UInt8Quantization(model=path_model, data_images=data)
    # convert_model2tflite_Int16Quantization(model=path_model, data_images=data)

    # Check file model tflite
    interpreter = tf.lite.Interpreter(model_path="person_detection.tflite")
    interpreter.allocate_tensors()

    print(interpreter.get_input_details()[0]['shape'])
    print(interpreter.get_input_details()[0]['dtype'])

    print(interpreter.get_output_details()[0]['shape'])
    print(interpreter.get_output_details()[0]['dtype'])
