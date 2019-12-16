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

df = pd.read_csv(folder + "\\titanic.csv")
#df = pd.read_csv('golf.csv', sep = '\t')                # read database


##def fun_entropy( str, log = 2 ):
##    count = 0
##    tempo = []
##    for i in range (len(df[str].unique())) :
##        tempo.append((df[str].value_counts()[i]) / len(df[str]))
##        #print('value ', i, ': ', tempo[i])
##        count += tempo[i] * math.log(tempo[i], log)
##    return (-count)


lb = LabelEncoder()                                         # Copy and transform value (ex: Str to int)


X = df[['pclass', 'sex', 'age', 'sibsp', 'parch',]]                                # Get Features table
y = df[['survived']]                                      # Get Target table

X = X.apply(lb.fit_transform)
y = y.apply(lb.fit_transform)

X_train, X_test , y_train,y_test = train_test_split(    # split your database and get Train data (70%) and Test data (30%)
    X, y, test_size = .3, random_state = 100)


print ("\n X: ", X)

tree = DecisionTreeClassifier(criterion = "entropy", max_depth=4)    # Generate your tree

tree.fit(X_train, y_train)                              # "rpart" generatte tree

y_pred = tree.predict(X_test)

print ("prediction : \n", y_pred)

print ("default value : \n", y_test)

print ("confusion: " ,confusion_matrix(y_test, y_pred))

print( "Accuracy : ", accuracy_score(y_test,y_pred)*100) 
#classification_report(y_test, y_pred)


dot_data = export_graphviz(                           # Create dot data
    tree, filled=True, rounded=True,
    feature_names= X.columns,
    out_file=None,
)

graph = pydotplus.graph_from_dot_data(dot_data)     # Create graph from dot data
graph.write_png('tree.png')                         # Write graph to PNG image
