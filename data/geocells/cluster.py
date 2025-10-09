from sklearn.cluster import OPTICS
import numpy as np
import matplotlib.pyplot as plt


def cluster(cell):
    X = []
    for point in cell.points:
        X.append(np.array([point["lng"], point["lat"]]))

    # for i in range(100):
    #     X.append(np.array([random.random(), random.random()]))
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

    new_cells = {}

    for i in range(len(labels)):
        if labels[i] not in new_cells:
            new_cells[int(labels[i])] = []
        new_cells[int(labels[i])].append(cell.points[i])

    return new_cells
