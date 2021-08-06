import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

with open('D:/Project/voice_recognization/phan_tich_ngon_ngu/demo/text_classify/Chatbot_Keras-main/intents_viet.json', encoding="utf-8") as file:
    data = json.load(file)


def chat():
    # load trained model
    model = keras.models.load_model("D:/Project/voice_recognization/phan_tich_ngon_ngu/demo/text_classify/Chatbot_Keras-main/chat_model_viet")

    # load tokenizer object
    with open('D:/Project/voice_recognization/phan_tich_ngon_ngu/demo/text_classify/Chatbot_Keras-main/tokenizer_viet.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # load label encoder object
    with open('D:/Project/voice_recognization/phan_tich_ngon_ngu/demo/text_classify/Chatbot_Keras-main/label_viet_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)

    # parameters
    max_len = 20
    
    while True:
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        if inp.lower() == "quit":
            break

        result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
        tag = lbl_encoder.inverse_transform([np.argmax(result)])

        for i in data['intents']:
            if i['tag'] == tag:
                print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , tag) #np.random.choice(i['responses']))

        # print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL,random.choice(responses))

print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
chat()
