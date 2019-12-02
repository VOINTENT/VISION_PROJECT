import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense, GlobalAveragePooling2D, Dropout, Flatten
from tensorflow.keras import optimizers
from tensorflow import keras

def get_model():
    model = Sequential()
    # Входной слой
    model.add(BatchNormalization(input_shape=(96, 96, 1)))
    model.add(Conv2D(24, (5, 5), kernel_initializer='he_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Слой 2
    model.add(Conv2D(36, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Слой 3
    model.add(Conv2D(48, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Слой 4
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.2))

    # Слой 5
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(Flatten())

    # Слой 6
    model.add(Dense(500, activation="relu"))

    # Слой 7
    model.add(Dense(90, activation="relu"))

    # Слой 8
    model.add(Dense(30))

    sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.95, nesterov=True)
    model.compile(optimizer=sgd, loss='mse', metrics=['accuracy'])

    return model

def load_model(FILENAME):
    return keras.models.load_model(FILENAME)

def fit_model(model, FILENAME, X_train, y_train, epochs=50):
    history = model.fit(X_train, y_train,
                 validation_split=0.2, shuffle=True,
                 epochs=epochs, batch_size=20)
    model.save(FILENAME)
    return history
