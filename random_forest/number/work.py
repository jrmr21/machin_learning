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
from sklearn.datasets import load_digits

print ('\n  ************** PICTURE ************** \n ')




##plt.gray() 
##plt.matshow(digits.images[0]) 
##plt.show()

##pprint(digits.images)

        ## ALL TARGET 
##x = 0
##for x in range(0, len(digits.target)) :
##    print ("- ", digits.target[x])
##    x += 1

digits = load_digits()

##n_samples = len(digits.images)
##
##X, y = digits.images, digits.target
##X = X.reshape((n_samples, -1))
X = digits['data']
y = digits['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify = y, test_size = 0.3)

# creation d'une foret
forest = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=11, min_samples_split=2,
                                min_samples_leaf=3, max_features='auto', max_leaf_nodes=None, bootstrap=True,
                                oob_score=True, random_state = 12)

# Apprentissage de cette foret
rfFit = forest.fit(X_train, y_train)

print ("features : ", len(X))

# Calculer l'erreur en OOB (en apprentissage)
print("score TRAIN oob: ", rfFit.oob_score_)

# Calculer l'erreur de prévision sur le test
print("score TEST oob: ", rfFit.score(X_test, y_test))

print('Train : ', rfFit.score(X_train, y_train))
print('TEST :  ', rfFit.score(X_test, y_test))



# creation d'une foret
##gs = GridSearchCV(
##        estimator=RandomForestRegressor(max_depth=14, n_estimators=100 ),
##        param_grid={
##            'max_depth': (3,4,5,6,7,8,9,10,11,12,13,14, 15),
##            'n_estimators': (10, 30, 50, 100, 200, 400, 600, 800, 1000),
##        }, cv=10, n_jobs=-1, scoring='neg_mean_squared_error'
##    )

gs = RandomForestRegressor(max_depth=14, n_estimators=100 )
# Apprentissage de cette foret
rfFit1 = gs.fit(X_train, y_train)

print(" \n\n REGRESSORT FOREST !!!!  ")

print('Train : ', rfFit1.score(X_train, y_train))
print('TEST :  ', rfFit1.score(X_test, y_test))

y_pred = gs.predict(X_test)

print('Best Params:')
print(gs.best_params_)
print('Best CV Score:')
#print(-gs.best_score_)

