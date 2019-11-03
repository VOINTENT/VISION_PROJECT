from imgaug.augmentables.kps import Keypoint, KeypointsOnImage
from imgaug.augmentables.heatmaps import HeatmapsOnImage

import imgaug.augmenters as iaa
import imageio
import imgaug as ia
import numpy as np

def get_keypoints(image):
    kps = [
        Keypoint(x=1459, y=1898),   # Ладонь

        Keypoint(x=907, y=1788),    # Большой
        Keypoint(x=911, y=1454),
        Keypoint(x=837, y=1104),

        Keypoint(x=1217, y=1347),   # Указательный
        Keypoint(x=1281, y=969),
        Keypoint(x=1349, y=386),

        Keypoint(x=1463, y=1383),   # Средний
        Keypoint(x=1485, y=993),
        Keypoint(x=1535, y=319),

        Keypoint(x=1679, y=1470),   # Безымянный
        Keypoint(x=1691, y=1114),
        Keypoint(x=1705, y=470),

        Keypoint(x=1881, y=1624),   # Мизинец
        Keypoint(x=1877, y=1324),
        Keypoint(x=1881, y=856)
    ]
    kpsoi = KeypointsOnImage(kps, shape=image.shape)
    return kpsoi

def get_seq():
    seq = iaa.Sequential([
        iaa.Affine(rotate=(-10, 10)),
        iaa.AddToHueAndSaturation((-50, 50))
    ])
    return seq

def get_heatmaps(kpsoi):
    # Формирование карт расстояний
    distance_maps = kpsoi.to_distance_maps()

    # Нормирование карт
    height, width = kpsoi.shape[0:2]
    max_distance = np.linalg.norm(np.float32([height, width]))
    distance_maps_normalized = distance_maps / max_distance

    # Инвертирование
    heatmaps = HeatmapsOnImage((1.0 - distance_maps_normalized)**10, shape=kpsoi.shape)

    return heatmaps

if __name__ == '__main__':
    ia.seed(3)

    image = imageio.imread("hand.jpg")
    kpsoi = get_keypoints(image)

    seq = get_seq()
    image_aug, kpsoi_aug = seq(image=image, keypoints=kpsoi)
    
    heatmaps = get_heatmaps(kpsoi)

    ia.imshow(ia.draw_grid(heatmaps.draw_on_image(image), cols=4, rows=4))
