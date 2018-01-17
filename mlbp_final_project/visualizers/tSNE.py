"""tSNE visualization edited from http://scikit-learn.org/stable/auto_examples/manifold/plot_lle_digits.html

Original authors:   Fabian Pedregosa <fabian.pedregosa@inria.fr>
                    Olivier Grisel <olivier.grisel@ensta.org>
                    Mathieu Blondel <mathieu@mblondel.org>
                    Gael Varoquaux
License: BSD 3 clause (C) INRIA 2011
"""


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import offsetbox
import numpy as np
import os
from sklearn import manifold
import time


import mlbp_final_project.utils.data_loader as data_loader


# Scale and visualize the embedding vectors
def plot_embedding(X, title=None):
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)
    for i in range(X.shape[0]):
        plt.text(X[i, 0], X[i, 1], y[i],
                 color=plt.cm.Set1(y[i]/10.0),
                 fontdict={'size': 2})

    # if hasattr(offsetbox, 'AnnotationBbox'):
    #     # only print thumbnails with matplotlib > 1.0
    #     shown_images = np.array([[1., 1.]])  # just something big
    #     for i in range(y.shape[0]):
    #         dist = np.sum((X[i] - shown_images) ** 2, 1)
    #         if np.min(dist) < 4e-3:
    #             # don't show points that are too close
    #             continue
    #         shown_images = np.r_[shown_images, [X[i]]]
    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


# t-SNE embedding of the digits dataset
def tSNE(X, y, title, show=False):

    random_state = int(time.time())
    print("Random state: {}".format(random_state))

    print("Computing t-SNE embedding")
    tsne = manifold.TSNE(n_components=3,
                         perplexity=500,
                         n_iter=5000,
                         init='pca',
                         random_state=random_state)

    X_tsne = tsne.fit_transform(X, y)
    print("tSNE shape: {}".format(X_tsne.shape))

    # plt.scatter(X_tsne[:,0], X_tsne[:,1], c=plt.cm.Set1(y[i]/10.0))
    # plot_embedding(X_tsne,
    #                "{} - time: {}".format(title, time.time() - t0))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # For each set of style and range settings, plot n random points in the box
    # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
    for i in range(len(y)):
        xs = X_tsne[i, 0]
        ys = X_tsne[i, 1]
        zs = X_tsne[i, 2]
        ax.scatter(xs, ys, zs, c=plt.cm.Set1(y[i]/10.0))

    ax.set_xlabel('x1 tSNE')
    ax.set_ylabel('x2 tSNE')
    ax.set_zlabel('x3 tSNE')

    if show:
        plt.show()
    else:
        filename = "visual_tSNE_{}_{}.png".format(title, random_state)
        plt.savefig(os.path.join(data_loader.get_output_folder(), filename))


if __name__ == "__main__":
    X = data_loader.load_train_data()[3000:4100, :]
    print(X)
    X = (X.T / np.mean(X, axis=1).T).T
    print(X)
    print(X.shape)
    y = data_loader.load_train_labels()[3000:4100]
    print(y.shape)
    tSNE(X, y, "main", show=False)
