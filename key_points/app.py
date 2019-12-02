import pandas as pd
import numpy as np
from tensorflow import keras

import data
import network as nt
import show

if __name__ == '__main__':
    FILENAME_NT = 'model.h5'
    FILENAME_TRAIN_DATA = 'data/training.csv'
    FILENAME_TEST_DATA = 'data/test.csv'

    df_train = pd.read_csv(FILENAME_TRAIN_DATA)
    df_test = pd.read_csv(FILENAME_TEST_DATA)

    X_train, y_train = data.get_train_data(df_train)
    X_test = data.get_test_data(df_test)

    model = nt.get_model()

    for i in range(10):
        image = X_test[i,:,:,0]
        show.show_predicted_image_with_keypoints(model, data.PIPELINE, image)

    # images = np.array([X_train[i,:,:,0] for i in range(10)]).reshape(10, 96, 96, 1)
    # show.show_predicted_images_with_keypoints(model, data.PIPELINE, images)

    # -------------------------------------------------------------------
    # history = nt.fit_model(model, FILENAME_NT, X_train, y_train)
    # model.save(FILENAME_NT)
    # -------------------------------------------------------------------
