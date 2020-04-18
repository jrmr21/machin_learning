

# w=10
# h=10
# fig=plt.figure(figsize=(8, 8))
# columns = 4
# rows = 5
# for i in range(1, columns*rows +1):
#     img = np.random.randint(10, size=(h,w))
#     fig.add_subplot(rows, columns, i)
#     plt.imshow(img)
# plt.show()




import os
import math

import pydotplus
from sklearn.tree import export_graphviz
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import pandas as pd
from pprint import pprint
import numpy  as np

import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import seaborn as sns

print ('\n  ************** PICTURE ************** \n ')

# get local directory
folder = "C:/Users/jerem/Documents/Git/machine_learning/random_forest/number/"

# lecture des datasets
df_train = pd.read_csv(folder + "/optdigits_train.csv", header = None)
df_test  = pd.read_csv(folder + "/optdigits_test.csv", header = None)

# split des datasets
X_test, y_test   = df_test[range(64)], df_test[64]
X_train, y_train = df_train[range(64)], df_train[64]


def show_number( number, tab_x, tab_y):
    img2 = tab_x.iloc[number]
    img2 = np.array(img2).reshape((8, 8))
    plt.title(f"L'image est : {tab_y[number]} à la ligne {number}")
    plt.imshow(img2)
    return

def show_number_predict( number, tab_x, tab_y, predict):
    img2 = tab_x.iloc[number]
    img2 = np.array(img2).reshape((8, 8))
    plt.title(f"L'image est : {tab_y[number]} et l'algo prédit {predict}")
    plt.imshow(img2)
    plt.show()
    return


# *************  Afficher les 5 premiers 7 de la tab df_test  *************

#   show_number_list (your number to show, number to show, tab X, tab Y, number of "class_search" to skip)
def show_number_list( class_search, nb_print, tab_x, tab_y, start = 0):
    i = 0      # 7 count number to 5
    j = 0      # search number
    while (i < (nb_print + start)) :
        if (tab_y[j] == class_search) :
            i += 1
            if (i > start) :
                print("count ", (i - start), "  | position in tab ", j)
                show_number( j, tab_x, tab_y)
        j += 1


def show_all_moy() :
    # show 6 picture moy

    img_moy_k = [np.mean(X_train[y_train == k], axis=0) for k in range(10)]
    img_moy_k = [np.array(i).reshape((8, 8)) for i in img_moy_k]

    fig     = plt.figure(figsize = (6, 6))
    columns = 5
    rows    = 2
    
    # print("size ", len(img_moy_k))

    x = 0
    for i in range(1, columns*rows + 1):
        fig.add_subplot(rows, columns, i)
        plt.imshow(img_moy_k[x])
        x += 1
    plt.show()


    # *****     main        ****** 
show_all_moy()
show_number_list( 7, 1, X_train, y_train)


# **************************    classification    **************************  
# creation d'une foret
forest = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=11, min_samples_split=2,
                                min_samples_leaf=3, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                oob_score=True, random_state = 12)

# Apprentissage de cette foret
rfFit = forest.fit(X_train, y_train)

print ("features : ", (len(y_test) + len(y_train)))

# Calculer l'erreur en OOB (en apprentissage)
print("score TRAIN oob: ", rfFit.oob_score_)

# Calculer l'erreur de prévision sur le test
print("score TEST oob: ", rfFit.score(X_test, y_test))

print('Train : ', rfFit.score(X_train, y_train))
print('TEST :  ', rfFit.score(X_test, y_test))



# **************************    régréssion    **************************  

# #recherche de parametres 
# gs = GridSearchCV(
#        estimator = RandomForestRegressor( max_depth=11, n_estimators=500),
#        param_grid = {
#            'max_depth': (3,4,5,6,7,8,9,10,11,12,13,14, 15),
#            'n_estimators': (10, 30, 50, 100, 200, 400, 600, 800, 1000),
#        }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
#    )

# #creation d'une foret aleatoire
# rfFit1 = RandomForestRegressor( max_depth=11, n_estimators=500, random_state = 12)

# # Apprentissage de cette foret avec les parametres optimals
# rfFit1.fit(X_train, y_train)

# print( "depth ", gs.estimator.max_depth, " estimator : ", gs.estimator.n_estimators)



# print(" \n\n REGRESSORT FOREST !!!!  ")

# print('Train : ', rfFit1.score(X_train, y_train))
# print('TEST :  ', rfFit1.score(X_test, y_test))
