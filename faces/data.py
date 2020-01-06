import PIL, glob, os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def get_images(directory: str = 'test_data') -> list:
    """
    Get images from the specified directory
    :param directory: directory name like 'test_data'
    :return: list of images of np.ndarray type
    """
    origin_path = os.getcwd()
    os.chdir(os.getcwd() + '/' + directory)

    images = []
    for file in glob.glob("*.jpg"):
        img = PIL.Image.open(file).convert('L')
        imgarr = np.array(img)
        imgarr = imgarr / 255
        images.append(imgarr)

    os.chdir(origin_path)
    return images


def show_images(images: np.ndarray):
    """
    Show images in the screen
    """
    for image in images:
        plt.figure()
        plt.imshow(image.reshape(96, 96), cmap="gray")
        plt.colorbar()
        plt.grid(False)
    plt.show()
