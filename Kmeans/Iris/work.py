#import des librairies l'environnement
import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets

print(" *****       Kmeans Iris     *****")

#chargement de base de données iris
iris = datasets.load_iris()


#affichage des données, vous permet de mieux comprendre le jeu de données (optionnel) 
# print(iris)
# print(iris.data)
# print(iris.feature_names)
# print(iris.target)
# print(iris.target_names)


    # get value data, target and convert to dadaframe
x = pd.DataFrame(iris.data, columns = ['Sepal_Length','Sepal_width','Petal_Length','Petal_width'])
y = pd.DataFrame(iris.target, columns = ['target'])


#Cluster K-means
model = KMeans(n_clusters = 3)

#calculate all cluster
model.fit(x)

# central point groups
centroids = model.cluster_centers_
#print(centroids)

print ("y: ", y, "\n")

# cluster show 
print(model.labels_)

# show cluster groups by 'petal length' and 'petal width'
plt.scatter(x.Petal_Length , x.Petal_width, s = 40)
plt.show()

print(x.columns)
colormap=np.array(['Red','green','blue', 'orange'])

plt.title('Classification réelle')
#                                                     PAS BIEN !!
#                                                        VV                         
plt.scatter(x.Petal_Length, x.Petal_width, c = colormap[iris.target], s=40)
plt.scatter(centroids[:, 2], centroids[:, 3], c='yellow', s=40)
plt.show()

plt.title('Classification K-means ')
plt.scatter(x.Petal_Length, x.Petal_width, c = colormap[model.labels_], s=40)
plt.scatter(centroids[:, 2], centroids[:, 3], c='yellow', s=40)
plt.show()
