import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Activation, MaxPooling2D, Dense, GlobalAveragePooling2D
from tensorflow.keras import optimizers
from tensorflow import keras

import data

def show_kepoints(df):
    """Показать пример хранения данных о keypoints"""
    keypoint_cols = list(df.columns)[:-1]
    xy = df.iloc[0][keypoint_cols].values.reshape((15, 2))
    print(xy)

def show_images(nrows=5, ncols=5):
    """Показать несколько случайных изображений из тренировочного датасета"""
    selection = np.random.choice(df.index, size=(nrows*ncols), replace=False)
    image_strings = df.loc[selection]['Image']
    fig, axes = plt.subplots(figsize=(10, 10), nrows=nrows, ncols=ncols)
    for string, ax in zip(image_strings, axes.ravel()):
        ax.imshow(string_to_numpy(string), cmap='gray')
        ax.axis('off')
    plt.show()

def show_image(image):
    """Отобразить изображение"""
    plt.imshow(image, cmap='gray')
    plt.show()

def show_image_with_keypoints(image, xy):
    """Отобразить изображение с keypoints"""
    plt.plot(xy[:, 0], xy[:, 1], 'ro')
    plt.imshow(image, cmap='gray')
    plt.show()

def show_predicted_image_with_keypoints(model, output_pipe, image):
    """Отобразить изображение с keypoints, предсказанное на моделе"""
    predictions = model.predict(image[np.newaxis, :, :, np.newaxis])
    xy_predictions = data.PIPELINE.inverse_transform(predictions).reshape(15, 2)
    show_image_with_keypoints(image.reshape(96, 96), xy_predictions)

def show_images_with_keypoints(df, nrows=5, ncols=5):
    """Показать несколько случайных изображений из тренировочного датасета с ключевыми точками"""
    selection = np.random.choice(df.index, size=(nrows*ncols), replace=False)
    image_strings = df.loc[selection]['Image']
    keypoint_cols = list(df.columns)[:-1]
    keypoints = df.loc[selection][keypoint_cols]
    fig, axes = plt.subplots(figsize=(10, 10), nrows=nrows, ncols=ncols)
    for string, (iloc, keypoint), ax in zip(image_strings, keypoints.iterrows(), axes.ravel()):
        xy = keypoint.values.reshape((15, 2))
        ax.imshow(string_to_numpy(string), cmap='gray')
        ax.plot(xy[:, 0], xy[:, 1], 'ro')
        ax.axis('off')
    plt.show()

def show_data_statistics(df):
    """Отобразить статистику наличия keypoints"""
    df.describe().loc['count'].plot.bar()
    plt.show()

def show_predicted_images_with_keypoints(model, output_pipe, images):
    """Отобразить изображения с keypoints, предсказанные на моделе"""
    fig, axes = plt.subplots(figsize=(10, 10), nrows=1, ncols=images.shape[0])
    for ind, ax in zip(range(images.shape[0]), axes.ravel()):
        img = images[ind, :, :, 0]
        predictions = model.predict(img[np.newaxis, :, :, np.newaxis])
        xy_predictions = output_pipe.inverse_transform(predictions).reshape(15, 2)
        ax.imshow(img, cmap='gray')
        ax.plot(xy_predictions[:, 0], xy_predictions[:, 1], 'bo')
        ax.axis('off')
    plt.show()

def show_graphics(history):
    # Точность
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # Функция потерь
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
