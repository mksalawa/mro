import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn import svm
from sklearn import preprocessing


def read_data():
    img = np.array(Image.open('data.bmp'))
    data = []
    classes = []
    for y in range(100):
        for x in range(100):
            pixel = img[y][x]
            # if pixel[0] == 253:
            #     img[y][x] = [250, 0, 0]
            # elif pixel[0] != 255:
            #     img[y][x] = [0, 0, 250]
            # i = Image.fromarray(img)
            # i.save('data_sm_conv.bmp')
            if pixel[0] != 255:
                data.append([x, y])
                if pixel[0] != 0:
                    classes.append(0)
                else:
                    classes.append(1)
    return np.array(data), np.array(classes)


def compute_margin_and_score(data, classes, c_vals, kernel, step):
    margins = []
    scores = []
    for c in c_vals:
        clf = svm.SVC(kernel=kernel, verbose=True, C=c)
        clf.fit(data, classes)
        w = clf.dual_coef_.dot(clf.support_vectors_)
        margins.append(2 / np.sqrt(np.sum((w ** 2))))
        scores.append(clf.score(data, classes) * 100)
        plt.clf()
        generate_chart(data, classes, kernel, c, clf, step)
    return scores, margins


def generate_final_chart(c_vals, scores, margins):

    wrong_class = {}
    for ker in scores.keys():
        wrong_class[ker] = (np.array(scores[ker]) - 100) * (-1)

    plt.clf()
    plt.scatter(c_vals, wrong_class['linear'], label='linear', marker='o', c='b')
    plt.scatter(c_vals, wrong_class['poly'], label='poly', marker='x', c='r')
    plt.scatter(c_vals, wrong_class['rbf'], label='rbf', marker=(5, 0), c='y')
    plt.ylim(0, 50)
    # plt.xlim(c_vals[0] - 5, c_vals[-1] + 5)
    plt.xlabel('C')
    plt.ylabel('% of improperly classified points')
    plt.legend(loc='best')
    plt.title('Percentage of improperly classified points for different C values')
    plt.savefig('charts/scores_all_improper.pdf')
    plt.savefig('charts/scores_all_improper.jpg')
    plt.clf()
    plt.figure()
    plt.scatter(c_vals, margins['linear'], label='linear', marker='o', c='b')
    plt.scatter(c_vals, margins['poly'], label='poly', marker='x', c='r')
    plt.scatter(c_vals, margins['rbf'], label='rbf', marker=(5, 0), c='y')
    # plt.ylim(np.min(margins) - 5, np.max(margins) + 5)
    # plt.xlim(c_vals[0] - 0.1, c_vals[-1] + 0.1)
    plt.xlabel('C')
    plt.ylabel('Margin width')
    plt.legend(loc='best')
    plt.title('Margin width for different C values')
    plt.savefig('charts/margins_all.pdf')
    plt.savefig('charts/margins_all.jpg')


def generate_chart(data, classes, k, c, clf, step):
    def predict(x):
        return clf.predict([x])

    x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
    y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
    Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
    Z = Z.reshape(XX.shape)
    plot_areas(predict, step, data)
    plot_2d_classes(data, classes, 'rb')
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-1, 0, 1])
    plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80, facecolors='none', zorder=10)
    plt.title('SVC (kernel=' + k + ', C=' + str(c) + ')')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('charts/' + k + '_' + str(c) + '.jpg')
    plt.savefig('charts/' + k + '_' + str(c) + '.pdf')


def plot_areas(predict, plot_step, X):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    Z = np.array([predict(x) for x in np.c_[xx.ravel(), yy.ravel()]])
    Z = Z.reshape(xx.shape)
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Spectral)
    return xx, yy, Z


def plot_2d_classes(X, y, colors):
    for i, c in zip(range(len(colors)), colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=c)


if __name__ == '__main__':
    # c_vals = np.arange(10, 31, 5)
    # c_vals = np.array([0.0001, 0.001, 0.005, 0.01, 0.05])
    c_vals = np.array([0.1, 0.5, 1, 5])
    larger = np.arange(10, 31, 5)
    c_vals = np.concatenate((c_vals, larger), axis=0)

    data, classes = read_data()
    scores, margins = {}, {}

    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data, classes)
    step = 0.05
    for k in ['linear', 'poly', 'rbf']:
        scores[k], margins[k] = compute_margin_and_score(data, classes, c_vals, k, step)

    print("\nscores:", scores)
    print("margins:", margins)
    generate_final_chart(c_vals, scores, margins)
