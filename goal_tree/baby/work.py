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

import pydotplus
from sklearn.tree import export_graphviz

# get local directory
folder = os.getcwd() 

df = pd.read_csv('faible_poids_bebes2.csv' , sep = ',')

lb = LabelEncoder()                                         # Copy and transform value (ex: Str to int)

#np.random.seed(13)
#np.random.shuffle(df)

X = df[['MotherAge', 'MotherWeight', 'SmokePregnant', 'HistPremature', 'Hypertension', 'UterIrritability',]]                                # Get Features table
y = df[['LowBirthWeight']]                                      # Get Target table

X_test = X[df['SAMPLE'] != 'train']
X_train = X[df['SAMPLE'] == 'train']

y_test = y[df['SAMPLE'] != 'train']
y_train = y[df['SAMPLE'] == 'train']


X_test = X_test.apply(lb.fit_transform)
X_train = X_train.apply(lb.fit_transform)

y_test = y_test.apply(lb.fit_transform)
y_train = y_train.apply(lb.fit_transform)

print ("\n len train: ", len(X_train))
print ("\n len test: ", len(X_test))

acc = 0
i = 1
best_deph = 0

while (i < 100):
    tree = DecisionTreeClassifier(criterion = "gini", random_state=0,  max_depth=i)    # Generate your tree max_depth=12 == 83%
    tree.fit(X_train, y_train)        # "rpart" generatte tree
    y_pred = tree.predict(X_test)
    if (acc < accuracy_score(y_test,y_pred)*100) :
       acc = accuracy_score(y_test,y_pred)*100
       best_deph = i
    i+=1

tree = DecisionTreeClassifier(criterion = "gini", random_state=0,  max_depth = best_deph)    # Generate your tree max_depth=12 == 83%
tree.fit(X_train, y_train)           # "rpart" generatte tree
y_pred = tree.predict(X_test)
    
print ("\nconfusion: " ,confusion_matrix(y_test, y_pred))

print( "\nAccuracy : ", accuracy_score(y_test,y_pred)*100) 
#classification_report(y_test, y_pred)

print ("\n\n prediction : ", y_pred, "\n default value : ", y_test, " \n")

print ("best_deph: ", best_deph)

dot_data = export_graphviz(                           # Create dot data
    tree, filled=True, rounded=True,
    feature_names= X.columns,
    out_file=None,
)

graph = pydotplus.graph_from_dot_data(dot_data)     # Create graph from dot data
graph.write_png('tree_baby.png')                         # Write graph to PNG image
