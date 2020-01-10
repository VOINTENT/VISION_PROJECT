from utils import DirectContext

import json, glob, csv
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


def resize_images_with_labels(new_size: tuple = (96, 96),
                              json_directory: str = 'data/json',
                              image_directory: str = 'data/images',
                              resized_json_directory: str = 'data/resized_json',
                              resized_image_directory: str = 'data/resized_images'):
    """
    Read json files and images, resize them and save to the new directories. Json and image files must have the same
    name like '1.json' and '1.jpg'
    """

    images = {}
    with DirectContext(image_directory):
        for file in glob.glob("*.jpg"):
            img = Image.open(file).convert('L')
            imgarr = np.array(img)  # imgarr = cv2.imread()
            images[file.split('.')[0]] = imgarr

    labels = {}
    with DirectContext(json_directory):
        for file in glob.glob("*.json"):
            with open(file, 'r') as f:
                data = json.load(f)

                label = {
                    'list_names': [],
                    'list_x': [],
                    'list_y': []
                }
                for shape in data['shapes']:
                    label['list_names'].append(shape['label'])
                    label['list_x'].append(shape['points'][0][0])
                    label['list_y'].append(shape['points'][0][1])

                assert len(label['list_x']) == len(label['list_y']) == 21
                labels[file.split('.')[0]] = label

    assert len(labels) == len(images)

    for name, imgarr in images.items():
        height, width = imgarr.shape

        height_coeff_resize = height / new_size[0]
        width_coeff_resize = width / new_size[1]

        image = Image.fromarray(imgarr).resize(new_size)
        imgarr = np.array(image)

        label = labels[name]
        for i in range( len( label['list_x'] ) ):
            label['list_x'][i] /= width_coeff_resize
            label['list_y'][i] /= height_coeff_resize

        with DirectContext(resized_image_directory):
            image.save(f'{name}.png')

        with DirectContext(resized_json_directory):
            with open(f'{name}.json', 'w') as f:
                json.dump(label, f, indent=4)


def resize_images(new_size: tuple = (96, 96),
                  image_directory: str = 'data/test_images',
                  resized_image_directory: str = 'data/test_images'):
    """
    Read images, resize them and save into new directory
    """
    images = {}
    with DirectContext(image_directory):
        for file in glob.glob("*.jpg"):
            img = Image.open(file).convert('L')
            imgarr = np.array(img)
            images[file.split('.')[0]] = imgarr

    with DirectContext(resized_image_directory):
        for name, imgarr in images.items():
            image = Image.fromarray(imgarr).resize(new_size)
            image.save(f'{name}.png')


def save_images_with_key_points(json_directory: str = 'data/resized_json',
                                image_directory: str = 'data/resized_images',
                                tagret_directory: str = 'data/images_with_keypoints',
                                ):
    """
    Read image and json files, save images with keypoints into new directory
    """
    images = {}
    with DirectContext(image_directory):
        for file in glob.glob(r'*.png'):
            img = Image.open(file).convert('L')
            imgarr = np.array(img)
            images[file.split('.')[0]] = imgarr

    labels = {}
    with DirectContext(json_directory):
        for file in glob.glob("*.json"):
            with open(file, 'r') as f:
                label = json.load(f)
                labels[file.split('.')[0]] = label

    with DirectContext(tagret_directory):
        for name, imgarr in images.items():
            plt.figure()
            plt.imshow(imgarr, cmap='gray')
            plt.colorbar()
            plt.grid(False)
            plt.scatter(labels[name]['list_x'], labels[name]['list_y'], c='red', s=12)
            plt.savefig(f'{name}.png')


def generate_training_csv(json_directory: str = 'data/resized_json',
                          image_directory: str = 'data/resized_images',
                          csv_directory: str = 'data/csv'):
    """
    Read json files and images, converts them to csv file 'training'. Json and image files
    must have the same name like '1.json' and '1.jpg'
    """
    images = {}
    with DirectContext(image_directory):
        for file in glob.glob(r'*.png'):
            img = Image.open(file).convert('L')
            imgarr = np.array(img)
            images[file.split('.')[0]] = imgarr

    labels = {}
    with DirectContext(json_directory):
        for file in glob.glob("*.json"):
            with open(file, 'r') as f:
                label = json.load(f)
                labels[file.split('.')[0]] = label

    assert len( labels ) == len( images )

    lines = [['1_x', '1_y', '2_x', '2_y', '3_x', '3_y', '4_x', '4_y', '5_x', '5_y', '6_x', '6_y', '7_x', '7_y',
              '8_x', '8_y', '9_x', '9_y', '10_x', '10_y', '11_x', '11_y', '12_x', '12_y', '13_x', '13_y', '14_x', '14_y',
              '15_x', '15_y', '16_x', '16_y', '17_x', '17_y', '18_x', '18_y', '19_x', '19_y', '20_x', '20_y', '21_x', '21_y',
              'Image']]

    for name, label in labels.items():

        line = []
        for x, y in zip(label['list_x'], label['list_y']):
            line.append(x)
            line.append(y)

        image_list = list(images[name].reshape((1, 9216))[0])
        image_str = ' '.join(map(str, image_list))
        line.append(image_str)

        lines.append(line)

    with DirectContext(csv_directory):
        with open('training.csv', "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(lines)


def generate_test_csv(image_directory: str = 'data/test_images',
                      csv_directory: str = 'data/csv'):
    """
    Read images, convert them to csv file 'test'
    :param image_directory:
    :param csv_directory:
    :return:
    """
    images = {}
    with DirectContext(image_directory):
        for file in glob.glob(r'*.png'):
            img = Image.open(file).convert('L')
            imgarr = np.array(img)
            images[file.split('.')[0]] = imgarr

    lines = [['ImageId', 'Image']]

    i = iter( range( len( images ) ) )
    for (name, imgarr) in images.items():
        image_list = list(images[name].reshape((1, 9216))[0])
        image_str = ' '.join(map(str, image_list))

        line = [next(i), image_str]
        lines.append(line)

        with DirectContext(csv_directory):
            with open('test.csv', "w", newline="") as file:
                writer = csv.writer(file)
                writer.writerows(lines)


if __name__ == '__main__':
    generate_test_csv()
    generate_training_csv()
