from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import load_model
from tensorflow.keras import optimizers
import tensorflow.keras
import numpy as np

x = np.load("Xtrain.npy")
y = np.load("Ytrain.npy")

num_hidden = 2
model = Sequential()
model.add(Dense(128, input_shape=(x.shape[1], x.shape[2])))
model.add(Activation("relu"))
for _ in range(num_hidden):
    model.add(Dense(128))
    model.add(Activation("relu"))

model.add(GRU(128))
model.add(Dense(2, activation = "softmax"))

adam = optimizers.Adam(lr=0.000125)
model.compile(optimizer=adam, loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

from sklearn.model_selection import train_test_split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.1, random_state=42)

model.fit(x_train, y_train, validation_data=(x_valid, y_valid), batch_size=32, epochs=12, shuffle=True)
model.save('model.h5')


