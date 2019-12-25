#import des librairies l'environnement
import os
import pandas as pd
import numpy as np
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import preprocessing

print(" *****       Kmeans voiture     *****")

#chargement de base de données iris
# get local directory
folder = os.getcwd() 

df = pd.read_csv("c:/Users/jrmr/Desktop/machine_learning/Kmeans/Nouveau dossier/voiture.txt" , sep = "\t")

print(df.head())
## get value data, target and convert to dadaframe

# Get Features table
X = pd.DataFrame(df[['mpg', 'displacement', 'horsepower', 'weight', 'acceleration']])

print("X tab: \n", X, " \n head de X \n ", X.horsepower)

#tout mettre a la meme échelle
X_cr = preprocessing.scale(X)

print("cr \n ", X_cr)

# Get Target table
#y = df[['origin']]
y = pd.DataFrame(df['origin'], columns = ['target'])

#Cluster K-means
model = KMeans(n_clusters = 4)

#calculate all cluster
model.fit(X_cr)

# central point groups
centroids = model.cluster_centers_
#print(centroids)

print ("y: ", y, "\n")

# cluster show 
print(model.labels_)

# show cluster groups by 'petal length' and 'petal width'
plt.scatter(X_cr[2] , X_cr[4], c = colormap[y], s = 40)
plt.show()

colormap=np.array(['Red','green','blue', 'orange'])

# plt.title('Classification réelle')                    
#plt.scatter(X.horsepower , X.acceleration, c = colormap[y], s=40)
# plt.scatter(centroids[:, 2], centroids[:, 3], c='yellow', s=40)
#plt.show()

# plt.title('Classification K-means ')
# plt.scatter(x.Petal_Length, x.Petal_width, c = colormap[model.labels_], s=40)
# plt.scatter(centroids[:, 2], centroids[:, 3], c='yellow', s=40)
# plt.show()
