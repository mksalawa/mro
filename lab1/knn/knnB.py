import random

import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial as sp
from sklearn import datasets

from lab1.knn import visualisation_util as vis_util


def get_iris_2d_data():
    iris = datasets.load_iris()
    data = iris.data[:, :2]  # we only take the first two features
    classes = iris.target

    processed_data = []
    processed_classes = []
    seen = set()
    for i, d in enumerate(data):
        if (d[0], d[1]) not in seen:
            processed_data.append(d)
            processed_classes.append(classes[i])
            seen.add((d[0], d[1]))
    return np.array(processed_data), np.array(processed_classes)


def uniform_weight(dist):
    return 1.


def distance_weight(dist):
    return 1. / dist


class KNN:
    def __init__(self, data, classes, k, weight_fun=uniform_weight):
        self.data = data
        self.classes = classes
        self.k = k
        self.weight_fun = weight_fun

    def neighbours(self, sample):
        dists = sp.distance.cdist([sample], self.data, 'euclidean').flatten()
        idx = np.argsort(dists)
        # returning k neighbours starting from the second (the first is self)
        idx = idx[:self.k]
        return idx, [self.weight_fun(dists[i]) for i in idx]

    def predict(self, sample):
        neigh_idx, neigh_weights = self.neighbours(sample)
        weighted_votes = {}
        for i, w in zip(neigh_idx, neigh_weights):
            c = self.classes[i]
            if c not in weighted_votes:
                weighted_votes[c] = 0.
            weighted_votes[c] += w
        return max(weighted_votes, key=weighted_votes.get)


class KNNWithCNN:
    def __init__(self, data, classes, k, weight_fun=uniform_weight):
        self.data = data
        self.classes = classes
        self.k = k
        self.weight_fun = weight_fun
        self.cnn, self.cnn_classes = self.compute_CNN_set()

    def compute_CNN_set(self):
        cnn = []
        cnn_classes = []
        added = False
        cnn.append(self.data[0])
        cnn_classes.append(self.classes[0])
        while not added:
            added = False
            for i in range(1, len(self.data)):
                d = self.data[i]
                predicted = self.predict_using_data(d, cnn, cnn_classes)
                if predicted != self.classes[i]:
                    cnn.append(d)
                    cnn_classes.append(self.classes[i])
                    added = True
        return np.array(cnn), np.array(cnn_classes)

    def predict_using_data(self, sample, data, classes):
        neigh_idx, neigh_weights = self.neighbours(sample, data, self.k, self.weight_fun)
        weighted_votes = {}
        for i, w in zip(neigh_idx, neigh_weights):
            c = classes[i]
            if c not in weighted_votes:
                weighted_votes[c] = 0.
            weighted_votes[c] += w
        return max(weighted_votes, key=weighted_votes.get)

    def neighbours(self, sample, data, k, weight_fun):
        dists = sp.distance.cdist([sample], data, 'euclidean').flatten()
        idx = np.argsort(dists)
        # returning k neighbours starting from the second (the first is self)
        idx = idx[:k]
        return idx, [weight_fun(dists[i]) for i in idx]

    def predict(self, sample):
        return self.predict_using_data(sample, self.cnn, self.cnn_classes)


def generate_chart(cnn, weighted, k):
    voting_text = "głosowanie z wagami 1/d" if weighted else "zwykłe głosowanie"
    comp_text = "kompresja CNN" if cnn else "brak CNN"

    filename = "charts/B/"
    if cnn:
        filename += "cnn-"
    if weighted:
        filename += "weighted-"
    filename += str(k) + ".pdf"
    plt.title("Klasyfikacja k-NN dla zbioru \"Iris\"\nodległosc Euklidesa, " + voting_text + ", " + comp_text + ", k=" + str(k))
    plt.xlabel("Długosc działki kielicha")
    plt.ylabel("Szerokosc działki kielicha")
    plt.savefig(filename)
    plt.clf()


def iris_B1(cnn, weighted, k):
    X, y = get_iris_2d_data()

    w_fun = uniform_weight
    if weighted:
        w_fun = distance_weight

    if cnn:
        shuffled_X = []
        shuffled_y = []
        ran = list(range(0, len(X)))
        random.seed(10000147)
        random.shuffle(ran)
        for i in ran:
            shuffled_X.append(X[i])
            shuffled_y.append(y[i])
        X = shuffled_X
        y = shuffled_y

        knn = KNNWithCNN(X, y, k, w_fun)
        print(len(knn.cnn) * 100 / len(X))
        X_to_plot = knn.cnn
        y_to_plot = knn.cnn_classes
    else:
        knn = KNN(X, y, k, w_fun)
        X_to_plot = X
        y_to_plot = y

    vis_util.plot_areas(knn.predict, 0.1, X_to_plot)
    vis_util.plot_2d_classes(X_to_plot, y_to_plot, 'ryb')

    generate_chart(cnn, weighted, k)


if __name__ == '__main__':
    # no CNN
    iris_B1(False, False, 1)
    iris_B1(False, True, 1)
    iris_B1(False, False, 5)
    iris_B1(False, True, 5)

    # with CNN
    iris_B1(True, False, 1)
    iris_B1(True, True, 1)
    iris_B1(True, False, 5)
    iris_B1(True, True, 5)


