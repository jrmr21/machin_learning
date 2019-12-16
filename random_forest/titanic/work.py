import os
import math

import pydotplus
from sklearn.tree import export_graphviz
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import pandas as pd
from pprint import pprint
import numpy  as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# get local directory
folder = os.getcwd() 
df = pd.read_csv(folder + "\\titanic.csv")

print ('\n  ************** TITANIC ************** \n ')
lb = LabelEncoder()                                         # Copy and transform value (ex: Str to int)


X = df[['pclass', 'sex', 'age', 'sibsp', 'parch',]]       # Get Features table
y = df[['survived']]                                      # Get Target table

X = X.apply(lb.fit_transform)
y = y.apply(lb.fit_transform)



X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.4)


# creation d'une foret
forest = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=None, min_samples_split=2,
                                min_samples_leaf=1, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                oob_score=True, random_state = None)

# Apprentissage de cette foret
rfFit = forest.fit(X_train, y_train)

# Calculer l'erreur en OOB (en apprentissage)
print("score TRAIN oob: ", rfFit.oob_score_)

# Calculer l'erreur de prévision sur le test
print("score TEST oob: ", rfFit.score(X_test, y_test))


# Optimisation par validation croisée du nombre de variables tirés aléatoirement lors de la construction de chaque noeud.
param = [ { "max_features" : list(range(2, 4, 3)) }, {'n_estimators': [120, 100, 140,500,700,1000]}, {'max_depth': [3,5,7,9,10,11,12,13,15,19]}]

rf = GridSearchCV(RandomForestClassifier(criterion='gini',  random_state = 0),
			        param, cv=5, n_jobs = -1)

# Paramètre optimal
rfOpt = rf.fit(X_train, y_train)
print("Meilleur score = %f, Meilleur paramètre = %s" % (1. - rfOpt.best_score_,rfOpt.best_params_))


print('Accuracy of RF classifier on training set: {:.2f}'
     .format(rfOpt.score(X_train, y_train)))
print('Accuracy of RF classifier on test set: {:.2f}'
     .format(rfOpt.score(X_test, y_test)))
