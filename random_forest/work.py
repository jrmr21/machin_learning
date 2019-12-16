# En comparaison avec R, le randomForest en Python comporte moins d'options mais l'utilisation de base est très similaire avec le même jeu de paramètres.
# On se concentre sur l'Optimisation du max_features

#https://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_iris.html
#sphx-glr-auto-examples-ensemble-plot-forest-iris-py

import os
import math

import pydotplus
from sklearn.tree import export_graphviz
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import pandas as pd
import random
from pprint import pprint

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV



    # load data with iris type
iris = datasets.load_iris()


#pprint (iris)

    # get value data, target and convert to dadaframe
X = pd.DataFrame(iris.data, columns = iris.feature_names)
y = pd.DataFrame(iris.target, columns = ['target'])

print (X, '\n', y)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.3)


# Instantiation ; Signature et définition des paramètres par défaut
forest = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split = 2,
                                min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                oob_score=True, random_state = 0)


# Apprentissage
rfFit = forest.fit(X_train, y_train)

# Calculer l'erreur en OOB (en apprentissage)
print("score TRAIN oob: ", rfFit.oob_score_)

# Calculer l'erreur de prévision sur le test
print("score TEST oob: ", rfFit.score(X_test, y_test))


# Optimisation par validation croisée du nombre de variables tirés aléatoirement lors de la construction de chaque noeud.
# Rq : On cherche à varier "max_features" pr trouver la val optimal !
param = [ { "max_features" : list(range(2, 4, 1)) } ]
rf = GridSearchCV(RandomForestClassifier(n_estimators=500, criterion='gini', random_state = 13),
			        param, cv=5, n_jobs = -1)

# Paramètre optimal
rfOpt = rf.fit(X_train, y_train)
print("Meilleur score = %f, Meilleur paramètre = %s" % (1. - rfOpt.best_score_,rfOpt.best_params_))

print('Accuracy of RF classifier on training set: {:.2f}'
     .format(rfOpt.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(rfOpt.score(X_test, y_test)))

#estimator = forest.estimator_

export_graphviz((RandomForestClassifier)rfOpt.estimators_[2], 
                out_file='tree.dot', 
                feature_names = iris.feature_names,
                class_names = iris.target_names,
                rounded = True, proportion = False, 
                precision = 2, filled = True)

# Convert to png
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])


