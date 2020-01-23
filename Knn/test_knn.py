import matplotlib

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets

def show_knn_graph(iX, iy, iclf) :
    h = .02
    
    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA','#97e2ff'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00','#00AAFF'])


    # calculate min, max and limits
    x_min, x_max = iX[:, 0].min() - 1, iX[:, 0].max() + 1
    y_min, y_max = iX[:, 1].min() - 1, iX[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
    np.arange(y_min, y_max, h))

    # predict class using data and kNN classifier
    Z = iclf.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

    # Plot also the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title("3-Class classification (k = %i)" % (n_neighbors))
    plt.show()

n_neighbors = 6

# import some data to play with
iris = datasets.load_iris()

# prepare data
X = iris.data[:, :2]
y = iris.target

# we create an instance of Neighbours Classifier and fit the data.
clf = neighbors.KNeighborsClassifier(n_neighbors, weights='distance')
clf.fit(X, y)

show_knn_graph(X, y, clf)
