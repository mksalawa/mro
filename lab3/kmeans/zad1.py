import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def create_data():
    POINTS = 100
    cov = [[0.01, 0], [0, 0.01]]
    points = []
    for i in range(3):
        for j in range(3):
            mean = (i + 1, j + 1)
            points.append(np.random.multivariate_normal(mean, np.array(cov), POINTS))
    # for i in range(9):
    #     plot_2d_classes(points[i], np.zeros(POINTS), 'w')
    # plt.title('Original data points')
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.xlim(0, 4)
    # plt.ylim(0, 4)
    # plt.savefig('charts/data.pdf')
    return np.array(points).reshape((9 * POINTS, 2))


def get_random_centroids(data, n):
    x_min = np.min(data[:, :1], axis=0)
    x_max = np.max(data[:, :1], axis=0)
    y_min = np.min(data[:, 1:], axis=0)
    y_max = np.max(data[:, 1:], axis=0)
    print(x_min, x_max, y_min, y_max)
    xs = np.random.random(n) * (x_max - x_min) + x_min
    ys = np.random.random(n) * (y_max - y_min) + y_min
    return np.array([(a, b) for a, b in zip(*(xs, ys))])


def get_random_partition_centroids(data, n):
    shuffled = np.copy(data)
    np.random.shuffle(shuffled)
    l = len(shuffled)
    res = []
    for i in range(n):
        start = math.floor(i * l / n)
        end = math.floor((i + 1) * l / n) - 1
        d = shuffled[start:end]
        res.append(np.mean(d, axis=0))
    return np.array(res)


def quality():
    rand_means, forgy_means, rand_part_means, kmeans_means = [], [], [], []
    rand_std, forgy_std, rand_part_std, kmeans_std = [], [], [], []
    iterations = list([1, 5, 10])
    iterations.extend(list(range(50, 351, 50)))

    for iters in iterations:
        rand, forgy, rand_part, kmeans = [], [], [], []

        for i in range(10):
            data = create_data()
            # random
            centroids = get_random_centroids(data, 9)
            # plot_2d_classes(centroids, np.zeros(len(centroids)), 'r')
            result = KMeans(n_clusters=9, max_iter=iters, init=centroids, n_init=1, n_jobs=-1).fit(data)
            # plot_2d_classes(result.cluster_centers_, np.zeros(len(result.cluster_centers_)), 'ggggggggg')
            # plt.show()
            rand.append(silhouette_score(data, result.labels_))

            # Forgy
            result = KMeans(n_clusters=9, max_iter=iters, init='random', n_init=1, n_jobs=-1).fit(data)
            # plot_2d_classes(result.cluster_centers_, np.zeros(len(result.cluster_centers_)), 'ggggggggg')
            # plt.show()
            forgy.append(silhouette_score(data, result.labels_))

            # Random Partition
            centroids = get_random_partition_centroids(data, 9)
            result = KMeans(n_clusters=9, max_iter=iters, init=centroids, n_init=1, n_jobs=-1).fit(data)
            # plot_2d_classes(result.cluster_centers_, np.zeros(len(result.cluster_centers_)), 'ggggggggg')
            # plt.show()
            rand_part.append(silhouette_score(data, result.labels_))

            # k-means++
            result = KMeans(n_clusters=9, max_iter=iters, init='k-means++', n_init=1, n_jobs=-1).fit(data)
            # plot_2d_classes(result.cluster_centers_, np.zeros(len(result.cluster_centers_)), 'ggggggggg')
            # plt.show()
            kmeans.append(silhouette_score(data, result.labels_))

        rand_means.append(np.mean(rand))
        rand_std.append(np.std(rand))
        forgy_means.append(np.mean(forgy))
        forgy_std.append(np.std(forgy))
        rand_part_means.append(np.mean(rand_part))
        rand_part_std.append(np.std(rand_part))
        kmeans_means.append(np.mean(kmeans))
        kmeans_std.append(np.std(kmeans))

    plt.clf()
    plt.title('Clustering quality for different initialisation methods')
    plt.xlabel('Max iterations')
    plt.ylabel('Quality (Silhouette score)')
    plt.errorbar(iterations, rand_means, yerr=rand_std, fmt='o-', c='b', label='Random')
    plt.errorbar(iterations, forgy_means, yerr=forgy_std, fmt='o-', c='g', label='Forgy')
    plt.errorbar(iterations, rand_part_means, yerr=rand_part_std, fmt='o-', c='r', label='Random Partition')
    plt.errorbar(iterations, kmeans_means, yerr=kmeans_std, fmt='o-', c='y', label='k-means++')
    plt.legend(loc='best')
    plt.savefig('charts/quality.pdf')


if __name__ == '__main__':
    quality()
