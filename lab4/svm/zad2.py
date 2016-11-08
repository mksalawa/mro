import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


def classifiers_coop():
    data = np.loadtxt('data_banknote_authentication.txt', delimiter=",")
    first = data[:762]
    sec = data[762:]
    data = np.concatenate((first[:200], sec[:200]), axis=0)

    svm_linear = svm.SVC(kernel='linear')
    svm_poly = svm.SVC(kernel='poly')
    svm_rbf_c = svm.SVC(kernel='rbf', C=30)
    svm_rbf = svm.SVC(kernel='rbf')
    knn_uniform_1 = KNeighborsClassifier(1)
    knn_uniform_5 = KNeighborsClassifier(5)
    knn_weighted_1 = KNeighborsClassifier(1, weights='distance')
    knn_weighted_5 = KNeighborsClassifier(5, weights='distance')
    logistic_reg = LogisticRegression()
    bagging = BaggingClassifier()

    classifiers = [svm_linear, svm_poly, svm_rbf_c, svm_rbf, knn_uniform_1, knn_uniform_5, knn_weighted_1,
                   knn_weighted_5]
    clf_accuracies = [[] for _ in range(10)]

    for iter in range(20):
        train, test = train_test_split(data, train_size=0.05)
        X_train = train[:, :4]
        y_train = train[:, 4]
        X_test = test[:, :4]
        y_test = test[:, 4]
        for clf in classifiers:
            clf.fit(X_train, y_train)

        train_results = [[clf.predict([x])[0] for clf in classifiers] for x in X_train]
        test_results = [[clf.predict([x])[0] for clf in classifiers] for x in X_test]
        logistic_reg.fit(train_results, y_train)
        bagging.fit(train_results, y_train)

        for clf_idx, clf in enumerate(classifiers):
            clf_accuracies[clf_idx].append(sum(1 if int(clf.predict([el])[0]) == int(y_test[i]) else 0 for i, el in enumerate(X_test)))
        clf_accuracies[8].append(sum(1 if int(logistic_reg.predict([el])[0]) == int(y_test[i]) else 0 for i, el in enumerate(test_results)))
        clf_accuracies[9].append(sum(1 if int(bagging.predict([el])[0]) == int(y_test[i]) else 0 for i, el in enumerate(test_results)))

        for k in range(len(clf_accuracies)):
            clf_accuracies[k][iter] = clf_accuracies[k][iter] * 100 / len(test_results)

    means = [np.mean(clf_accuracies[i]) for i in range(len(clf_accuracies))]
    stds = [np.std(clf_accuracies[i]) for i in range(len(clf_accuracies))]

    n_groups = 10
    index = np.arange(n_groups)
    bar_width = 0.4
    labels = ['SVM linear', 'SVM poly C=1', 'SVM rbf C=30', 'SVM rbf', 'kNN uniform 1', 'kNN uniform 5',
              'kNN wieghted 1', 'kNN weighted 5', 'Logistical Regression', 'Bagging']
    plt.bar(index+bar_width, means, bar_width, yerr=stds, tick_label='Accuracy')
    plt.title('Classification accuracy depending on the classifier type')
    plt.xlabel('Classifier')
    plt.ylabel('Accuracy [%]')
    plt.xticks(index, labels, rotation=45)
    plt.tight_layout()
    plt.savefig('charts/ensamble_classifiers.jpg')
    plt.savefig('charts/ensamble_classifiers.pdf')
    plt.show()


if __name__ == '__main__':
    classifiers_coop()
