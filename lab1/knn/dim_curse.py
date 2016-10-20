import math
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial as sp

VECTOR_COUNT = 4000
CUBE_EDGE = 2
HYPERSPHERE_RADIUS = 1


def hypersphere_in_hypercube():

    def count_points_in_hs(dim):
        # generate VECTOR_COUNT vectors in hypercube of edge=CUBE_EDGE and given dimension, coords in [-1, 1)
        points = CUBE_EDGE * np.random.random_sample((VECTOR_COUNT, dim)) - 1
        norms = np.linalg.norm(points, axis=1)
        return sum(1 for d in norms if d <= HYPERSPHERE_RADIUS) * 100. / VECTOR_COUNT

    dimensions = range(2, 21)
    mean = []
    std_dev = []
    for dimension in dimensions:
        results = []
        for r in range(0, 10):
            results.append(count_points_in_hs(dimension))
        mean.append(np.mean(results))
        std_dev.append(np.std(results))

    plt.figure()
    plt.title('Ratio of hypersphere to hypercube volume (4000 samples)')
    plt.errorbar(dimensions, mean, yerr=std_dev, fmt='o')
    plt.ylabel('% of points inside hypersphere')
    plt.xlabel('Dimension')
    plt.ylim(0)
    plt.xlim(0)
    plt.savefig("charts/A1-20.pdf")
    plt.clf()


def hypercube_distances():
    dimensions = range(2, 201)
    VECTOR_COUNT = 100

    mean = []
    std_dev = []
    for dimension in dimensions:
        results = []
        for r in range(0, 10):
            pointsA = np.random.random_sample((VECTOR_COUNT, dimension))
            pointsB = np.random.random_sample((VECTOR_COUNT, dimension))
            distances = sp.distance.cdist(pointsA, pointsB, 'euclidean')
            results.append(np.std(distances) / np.mean(distances))
        mean.append(np.mean(results))
        std_dev.append(np.std(results))

    plt.figure()
    plt.title('Ratio of standard deviation to mean distance \nbetween random points (10000 sample distances)')
    plt.errorbar(dimensions, mean, yerr=std_dev, fmt='o')
    plt.ylabel('Standard deviation to mean distance ratio')
    plt.xlabel('Dimension')
    plt.ylim(0)
    plt.xlim(0)
    plt.savefig("charts/A2-200.pdf")
    plt.clf()


def hypercube_distances_dist():
    VECTOR_COUNT = 100
    interesting = [2, 5, 10, 20, 50, 100, 150, 200]
    for dimension in interesting:
        pointsA = np.random.random_sample((VECTOR_COUNT, dimension))
        pointsB = np.random.random_sample((VECTOR_COUNT, dimension))
        distances = sp.distance.cdist(pointsA, pointsB, 'euclidean')
        distances /= math.sqrt(dimension)

        plt.figure()
        plt.title('Normalized distance between points (10000 sample distances)')
        plt.hist(distances.flatten(), 20)
        plt.xlabel('Normalized distance')
        plt.ylabel('Frequency')
        plt.ylim(0)
        plt.xlim(0, 1)
        plt.savefig("charts/dists/A2dists-" + str(dimension) + ".pdf")
        plt.clf()


if __name__ == '__main__':
    hypersphere_in_hypercube()
    hypercube_distances()
