import network, data

import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    images = data.get_images('test_data')

    os.chdir(os.getcwd() + '/' + 'result_data')

    for i in range( len( images ) ):
        image = images[i]
        list_x, list_y = network.get_points(image)
        plt.figure()
        plt.imshow(image.reshape(96, 96), cmap="gray")
        plt.axis("off")
        plt.scatter(list_x, list_y, c='red', s=12)
        plt.savefig(f'{i}.png')
