import matplotlib.pyplot as plt
import numpy as np


def points_shapes_lines():
    mean = [0, 0]
    cov = [[100, 30], [10, 5]]
    x, y = np.random.multivariate_normal(mean, cov, 200).T
    plt.plot(x, y, 'x')
    mean2 = [0, 0]
    cov2 = [[100, 30], [10, 5]]
    x2, y2 = np.random.multivariate_normal(mean2, cov2, 200).T
    x2 *= -1
    plt.plot(x2, y2, '.')
    plt.axis('equal')
    plt.show()


def points_shapes_circles():
    mean = [0, 0]
    cov = [[100, 0], [0, 100]]
    m = np.random.multivariate_normal(mean, cov, 400)
    radius = np.mean(np.linalg.norm(m, axis=1)) * 5
    outer, inner = [], []
    for p in m:
        inner.append(p) if p[0]**2 + p[1]**2 < radius else outer.append(p)
    outer = np.array(outer)
    inner = np.array(inner)
    plt.plot(outer[:, 0], outer[:, 1], 'x')
    plt.plot(inner[:, 0], inner[:, 1], '.')
    plt.axis('equal')
    plt.show()


if __name__ == '__main__':
    # points_shapes_lines()
    points_shapes_circles()
