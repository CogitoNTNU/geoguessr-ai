from sklearn.cluster import OPTICS
import numpy as np
import matplotlib.pyplot as plt
import random


def cluster():
    X = []
    for i in range(100):
        X.append(np.array([random.random(), random.random()]))
    X = np.array(X)
    print(X)

    clustering = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.05)
    clustering.fit(X)

    labels = clustering.labels_

    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap="rainbow", edgecolor="k")
    plt.title("OPTICS Clustering on Synthetic Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()


cluster()
