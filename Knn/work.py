#########################################################################
#####       					 iris		    	 	 	 	 	#####
import pandas as pd
from sklearn import datasets
iris = datasets.load_iris()


#########################################################################
#####       					 train_test_split		    	 	#####
import numpy as np
from sklearn.model_selection import train_test_split

# create design matrix X and target vector y
X = pd.DataFrame(iris.data, columns = ['Sepal_Length','Sepal_width','Petal_Length','Petal_width'])
y = pd.DataFrame(iris.target, columns = ['target'])

from sklearn import preprocessing


# mis a niveau des valeurs (min-max) soit une normalisation
min_max_scaler = preprocessing.MinMaxScaler()
X = min_max_scaler.fit_transform(X)


# compression des données en deux colonnes
from sklearn.decomposition import PCA

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# Reduce dimension to 2 with PCA
pca = make_pipeline(StandardScaler(),
                    PCA(n_components=2, random_state=0))
X_r = pca.fit(X).transform(X)


# split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_r, y, test_size=0.3, random_state=0)


#########################################################################
#####       						    knn 	 	 	 	 	 	#####
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# instantiate learning model (k = 3)
knn = KNeighborsClassifier(n_neighbors=34, n_jobs = -1)

# convert Y_train to np.array
y_train = y_train.to_numpy()

print("\n\nX: ",type(X_train), "\nY: ", type(y_train))

# fitting the model
knn.fit(X_train, y_train.ravel())

knn.effective_metric_ 	# 'euclidean'
#print(knn.kneighbors()) 	# la liste des voisins


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
    plt.scatter(iX[:, 0], iX[:, 1], c=iy.ravel(), cmap=cmap_bold)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    #plt.title("3-Class classification (k = %i)" % (iclf))
    plt.show()

show_knn_graph(X_train, y_train, knn)


#########################################################################
#####       		Parameter Tuning with Cross Validation		    #####

from sklearn.model_selection import cross_val_score

# empty list that will hold cv scores
error = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train.ravel())
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test.values))


#print(error, "\n\n len all sample: ", len(X_train) + len(X_test) )

plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')

plt.show()