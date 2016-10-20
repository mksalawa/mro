import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from PIL import Image

from lab2.pca.visualisation_util import plot_2d_classes

WIDTH = 80
HEIGHT = 106


def load_images():
    images = []
    nums = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12', '13', '14', '15']
    for i in nums:
        images.append(np.array(Image.open('photos/original/M' + i + '.jpg')).ravel())
        images.append(np.array(Image.open('photos/original/G' + i + '.jpg')).ravel())
    return images


def medium_image(images):
    mean = np.mean(images, axis=0)
    plt.gray()
    plt.imshow(mean.reshape((HEIGHT, WIDTH)))
    plt.title('Mean image')
    plt.savefig('photos/mean.jpg')
    plt.clf()


def image_PCA(images):
    pca = PCA()
    pca.fit_transform(images)
    for i, v in enumerate(pca.components_):
        plt.gray()
        plt.imshow(v.reshape((HEIGHT, WIDTH)))
        plt.title('Component ' + str(i))
        plt.savefig('photos/components/' + str(i) + '.jpg')
        plt.clf()
    plt.plot(pca.explained_variance_ratio_, 'o')
    plt.title('Explained variance ratio')
    plt.xlabel('Component')
    plt.xlim(-1)
    plt.savefig('photos/variance_ratio.pdf')
    plt.clf()


def image_PCA_reduced(images, comp):
    pca = PCA(n_components=comp)
    results = pca.fit_transform(images)
    inversed = pca.inverse_transform(results)
    for i, v in enumerate(inversed):
        plt.gray()
        plt.imshow(v.reshape((HEIGHT, WIDTH)))
        plt.title('Inversed ' + _get_image_name(i) + ' image (PC reduced to ' + str(comp) + ')')
        plt.savefig('photos/inversed/' + str(comp) + '/' + str(i) + '.jpg')
        plt.clf()


def image_PCA_2d(images):
    pca = PCA(n_components=2)
    results = pca.fit_transform(images)
    classes = np.array([i % 2 for i in range(30)])
    plots = plot_2d_classes(results, classes, 'br')
    plt.title('2D projection of the images')
    plt.xlabel('PC0')
    plt.ylabel('PC1')
    plt.legend(plots, ('Mxx images', 'Gxx images'))
    plt.savefig('photos/2d.pdf')
    plt.clf()


def _get_image_name(idx):
    name = 'M' if idx % 2 == 0 else 'G'
    i = int(math.ceil((idx + 1) / 2.))
    name += '0' + str(i) if i < 10 else str(i)
    return name


if __name__ == '__main__':
    images = load_images()
    medium_image(images)
    image_PCA(images)
    image_PCA_reduced(images, 5)
    image_PCA_reduced(images, 15)
    image_PCA_reduced(images, 20)
    image_PCA_reduced(images, 30)
    image_PCA_2d(images)
