import matplotlib.pyplot as plt
import numpy as np


def plot_2d_classes(X, y, colors):
    plots = []
    for i, c in zip(range(len(colors)), colors):
        idx = np.where(y == i)
        plots.append(plt.scatter(X[idx, 0], X[idx, 1], c=c))
    return plots
