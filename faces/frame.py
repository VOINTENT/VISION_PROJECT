import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

import network

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
RED = (0, 0, 255)
RADIUS = 3

# Кнут искусство программирования

while True:
    ret, img = cap.read()

    # function img to points
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # TODO: Rename coordinates
        gray_image = gray[y:y+h, x:x+w]
        size = gray_image.shape[0]
        resize_coeff_to_96x96 = size / 96
        image = Image.fromarray(gray_image).resize((96, 96))
        imgarr = np.array(image)

        list_x, list_y = network.get_points(imgarr)

        # This actually gui
        for x_point, y_point in zip(list_x, list_y):
            cv2.circle(img, (x+int(x_point * resize_coeff_to_96x96), y+int(y_point * resize_coeff_to_96x96)), RADIUS, RED)

        plt.figure()
        plt.imshow(imgarr, cmap='gray')
        plt.colorbar()
        plt.grid(False)
        plt.scatter(list_x, list_y, c='red', s=12)
        plt.show()

        # break

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
