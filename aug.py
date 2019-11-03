from imgaug import augmenters as iaa

import imageio
import imgaug as ia
import numpy as np


def get_seq(augs: list):

    seq_augs = []
    if 'affine' in augs:
        # Афинные преобразования
        seq_augs.append(iaa.Affine(rotate=(-20, 20)))

    if 'gaussian_noise':
        # Гаусовский шум
        seq_augs.append(iaa.AdditiveGaussianNoise(scale=(10, 60)))

    if 'crop' in augs:
        # Увеличение с обрезанием
        seq_augs.append(iaa.Crop(percent=(0, 0.2)))

    if 'change_color' in augs:
        # Изменение цвета
        seq_augs.append(iaa.AddToHueAndSaturation((-60, 60)))

    if 'large_noise' in augs:
        # Зашумление крупных объектов
        seq_augs.append(iaa.CoarseDropout((0.01, 0.1), size_percent=0.01))

    seq = iaa.Sequential(seq_augs, random_order=True)
    return seq

if __name__ == '__main__':
    ia.seed(4)

    image = imageio.imread("hand.jpg")
    images = [image, image, image, image]

    seq = get_seq(['affine', 'gaussian_noise', 'crop'])
    images_aug = seq.augment_images(images)
    ia.imshow(np.hstack(images_aug))
