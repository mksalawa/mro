import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image


def image_kmeans():
    FILE_NAME = 'salvador'
    FILE = FILE_NAME + '.bmp'
    image = np.array(Image.open(FILE))
    HEIGHT = image.shape[0]
    WIDTH = image.shape[1]
    data = image.reshape((HEIGHT * WIDTH), 3)
    clusters = [8, 16, 32, 64]

    for cols in clusters:
        result = KMeans(n_clusters=cols, init='k-means++', n_init=10, n_jobs=-1).fit(data)
        centers = result.cluster_centers_
        labels = result.labels_
        recoloured = np.array([[int(math.fabs(256 - centers[l][0])),
                                int(math.fabs(256 - centers[l][1])),
                                int(math.fabs(256 - centers[l][2]))] for l in labels]).reshape((HEIGHT, WIDTH, 3))

        plt.imsave('charts/2/converted_' + str(cols) + FILE_NAME + '.png', recoloured)
        plt.imsave('charts/2/converted_' + str(cols) + FILE_NAME + '.jpg', recoloured)
        plt.clf()
        print('Done', cols)


def charts():
    jpg = [538438, 569080, 571287, 473032]
    clusters = [1751958, 1752006, 1752102, 0]
    bmp = [5256962]

    n_groups = 4
    index = np.arange(n_groups)
    bar_width = 0.3

    plt.bar(index, [i/1024 for i in jpg], bar_width, color='b', label='JPEG')
    plt.bar(index + bar_width, [i/1024 for i in clusters], bar_width, color='g', label='Clusters')
    plt.bar(3 + bar_width, [i/1024 for i in bmp], bar_width, color='r', label='BMP')

    plt.title('File sizes depending on the file format')
    plt.ylabel('File size [kB]')
    plt.xlabel('Number of clusters')
    plt.xticks(index + bar_width, ('16', '32', '64', 'original'))

    plt.legend(loc='best')
    plt.savefig('charts/B-sizes.pdf')
    plt.savefig('charts/B-sizes.jpg')
    plt.show()


if __name__ == '__main__':
    image_kmeans()
    charts()
