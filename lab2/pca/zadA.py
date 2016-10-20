import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn.decomposition import PCA, KernelPCA

from lab2.pca.visualisation_util import plot_2d_classes


def compute_kernelPCA(data, classes, base_file_name, kernel='linear', gamma=None):
    pca = KernelPCA(kernel=kernel, gamma=gamma)
    results = pca.fit_transform(data)
    plot_2d_classes(results, classes, 'br')

    plt.title('Points after applying kernel PCA (' + kernel + ')' + ('' if gamma is None else ', gamma=' + str(gamma)))
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(base_file_name + '_PCA_base.pdf')
    plt.clf()


def compute_PCA_with_vectors(data, classes, base_file_name):
    plot_2d_classes(data, classes, 'br')

    pca = PCA()
    results = pca.fit_transform(data)
    ax = plt.axes()
    ax.arrow(0, 0, pca.components_[0][0], pca.components_[0][1])
    ax.arrow(0, 0, pca.components_[1][0], pca.components_[1][1])
    plt.title('Original points with PC vectors')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(base_file_name + '_original.pdf')
    plt.clf()

    plt.title('Points after applying PCA')
    plt.xlabel('x')
    plt.ylabel('y')
    ax = plt.axes()
    ax.arrow(0, 0, pca.components_[0][0], pca.components_[0][1])
    ax.arrow(0, 0, pca.components_[1][0], pca.components_[1][1])
    plot_2d_classes(results, classes, 'br')
    plt.savefig(base_file_name + '_PCA.pdf')
    plt.clf()


def points_lines():
    POINTS = 100
    x = 20 * np.random.sample(POINTS) - 10
    y = [x_i * 0.6 + (2 * np.random.random() - 1) for x_i in x]
    data = np.array([(xi, yi) for xi, yi in zip(x, y)])
    x2 = 20 * np.random.sample(POINTS) - 10
    y2 = [x_i * 0.4 + (2 * np.random.random() - 1) for x_i in x2]
    x2 *= -1
    data2 = np.array([(xi, yi) for xi, yi in zip(x2, y2)])
    all = np.concatenate((data, data2), axis=0)
    classes = np.array([0 if i < POINTS else 1 for i in range(POINTS * 2)])

    compute_PCA_with_vectors(all, classes, 'charts/b')
    compute_kernelPCA(all, classes, 'charts/b_rbf', 'rbf', gamma=5)
    compute_kernelPCA(all, classes, 'charts/b_cosine', 'cosine')


def points_circles():
    POINTS = 200
    points, classes = make_circles(n_samples=POINTS * 2, factor=0.2, noise=0.05)

    compute_PCA_with_vectors(points, classes, 'charts/a')
    compute_kernelPCA(points, classes, 'charts/a_rbf', 'rbf', gamma=20)
    compute_kernelPCA(points, classes, 'charts/a_cosine', 'cosine')


if __name__ == '__main__':
    # points_circles()
    points_lines()
