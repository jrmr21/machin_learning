import os
import math

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

import pandas as pd
import numpy  as np

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

import pydotplus
from sklearn.tree import export_graphviz

from sklearn.model_selection import GridSearchCV


df_train    = pd.read_excel('C:/Users/jerem/Documents/Git/machine_learning/goal_tree/titanic_2/med/pima.xls',
                   sheet_name = "apprentissage")


df_test     = pd.read_excel('C:/Users/jerem/Documents/Git/machine_learning/goal_tree/titanic_2/med/pima.xls',
                   sheet_name = "a_classer")

df_Y_test   = pd.read_excel('C:/Users/jerem/Documents/Git/machine_learning/goal_tree/titanic_2/med/pima.xls',
                   sheet_name = "etiquette")


lb = LabelEncoder()                                         # Copy and transform value (ex: Str to int)


X_train = df_train[['pregnant', 'plasma', 'bodymass', 'pedigree', 'age']]        # Get Features table
X_test  = df_test[['pregnant', 'plasma', 'bodymass', 'pedigree', 'age']]        # Get Features table


y_train     = df_train[['diabete']]
y_test_resp = df_Y_test[['diabete']]


y_train     = y_train.apply(lb.fit_transform)
y_test_resp = y_test_resp.apply(lb.fit_transform)



custom_parameter = [
##  {'criterion':['entropy'],
##   'max_depth':[i for i in range(0,150)],
##   'max_features':[i for i in range(1, 5)],
##   'random_state':[i for i in range(5, 10)]},

  {'criterion':['gini'],
   'max_depth':[i for i in range(0,50)],
   'max_features':[i for i in range(1, 5)],
   'random_state':[i for i in range(5, 10)],
   'n_estimators':[i for i in range(1, 100)]},
]


tree = GridSearchCV(RandomForestClassifier(bootstrap=True),
                    param_grid = custom_parameter, n_jobs = -1)

# Paramètre optimal
tree.fit(X_train, y_train)


print("Meilleur score = %f, Meilleur paramètre = %s" % (1. - tree.best_score_, tree.best_params_))

y_pred = tree.best_estimator_.predict(X_test)



print ("confusion: " , confusion_matrix(y_test_resp, y_pred))
print( "Accuracy : ", accuracy_score(y_test_resp, y_pred) * 100)


dot_data = export_graphviz(                           # Create dot data
    tree.best_estimator_ , filled=True, rounded=True,
    feature_names= X_test.columns,
    out_file=None,
)

graph = pydotplus.graph_from_dot_data(dot_data)     # Create graph from dot data
graph.write_png('tree_diabete.png')                 # Write graph to PNG image
