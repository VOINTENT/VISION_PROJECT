import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

"""Получить приведение значений к виду (-1, 1) и обратно"""
PIPELINE = make_pipeline( MinMaxScaler(feature_range=(-1, 1)) )

def string_to_numpy(string):
    """Преобразовать строку в массив numpy"""
    return np.array( [int(item) for item in string.split()]).reshape((96, 96) )

def get_train_data(df):
    """Получить треннировочные данные"""
    fully_annotated = df.dropna()
    X = np.stack([string_to_numpy(string) for string in fully_annotated['Image']]).astype(np.float)[:, :, :, np.newaxis]
    y = np.vstack(fully_annotated[fully_annotated.columns[:-1]].values)

    X_train = X / 255.
    y_train = PIPELINE.fit_transform(y)

    return X_train, y_train

def get_test_data(df):
    """Получить треннировочные данные"""
    X = np.stack([string_to_numpy(string) for string in df['Image']]).astype(np.float)[:, :, :, np.newaxis]
    X_test = X / 255.
    return X_test
