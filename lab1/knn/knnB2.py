import random
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.cross_validation import train_test_split

from lab1.knn.knnB import KNN, KNNWithCNN, distance_weight, uniform_weight


def generate_results_chart(results):
    print(results)

    accuracy = results[:, 0:1]
    std_dev = results[:, 1:2]
    cnn_size = results[:, 2:3]
    cnn_std = results[:, 3:4]
    x = list(range(1, 9))
    labels = ["k=1, uniform", "k=1, distance", "k=5, uniform", "k=5, distance",
              "k=1, uniform, CNN", "k=1, distance, CNN", "k=5, uniform, CNN", "k=5, distance, CNN"]

    plt.figure()
    plt.xlim(0, 10)
    plt.ylim(0, 100)
    plt.errorbar(x, accuracy, yerr=std_dev, fmt="o", label="Answer accuracy [%]")
    plt.errorbar(x, cnn_size, yerr=cnn_std, fmt="o", label="Compressed data\n set size [%]")
    plt.xticks(x, labels, rotation='vertical')
    plt.title("Klasyfikacja k-NN dla zbioru \"Iris\"")
    plt.subplots_adjust(bottom=0.3, right=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig('charts/B/results.pdf')
    plt.show()


def iris_B2(cnn, weighted, k):
    iris = datasets.load_iris()
    data = iris.data
    classes = iris.target

    processed_data = []
    processed_classes = []
    one = np.argwhere(classes == 1)[0][0]
    two = np.argwhere(classes == 2)[0][0]
    processed_data.append(data[:one])
    processed_data.append(data[one:two])
    processed_data.append(data[two:])
    processed_classes.append(classes[:one])
    processed_classes.append(classes[one:two])
    processed_classes.append(classes[two:])

    w_fun = uniform_weight
    if weighted:
        w_fun = distance_weight

    results = []
    compression_results = []
    for _ in range(0, 50):
        X_train = [[], [], []]
        X_test = [[], [], []]
        Y_train = [[], [], []]
        Y_test = [[], [], []]
        for c in [0, 1, 2]:
            X_train[c], X_test[c], Y_train[c], Y_test[c] = train_test_split(processed_data[c], processed_classes[c],
                                                                            test_size=0.3)
        X_train = np.concatenate(X_train)
        X_test = np.concatenate(X_test)
        Y_train = np.concatenate(Y_train)
        Y_test = np.concatenate(Y_test)

        if cnn:
            shuffled_X = []
            shuffled_y = []
            ran = list(range(0, len(X_train)))
            random.shuffle(ran)
            for i in ran:
                shuffled_X.append(X_train[i])
                shuffled_y.append(Y_train[i])
            X_train = shuffled_X
            Y_train = shuffled_y
            knn = KNNWithCNN(X_train, Y_train, k, w_fun)
            compression_results.append(len(knn.cnn) * 100 / len(X_train))
        else:
            knn = KNN(X_train, Y_train, k, w_fun)
        results.append(sum(1 for d, c in zip(X_test, Y_test) if c == knn.predict(d)) * 100. / len(X_test))

    return np.mean(results), np.std(results), \
           np.mean(compression_results) if cnn else 0, np.std(compression_results) if cnn else 0


if __name__ == '__main__':
    res = list()
    # no CNN
    res.append(list(iris_B2(False, False, 1)))
    res.append(list(iris_B2(False, True, 1)))
    res.append(list(iris_B2(False, False, 5)))
    res.append(list(iris_B2(False, True, 5)))

    # with CNN
    res.append(list(iris_B2(True, False, 1)))
    res.append(list(iris_B2(True, True, 1)))
    res.append(list(iris_B2(True, False, 5)))
    res.append(list(iris_B2(True, True, 5)))

    generate_results_chart(np.array(res))
