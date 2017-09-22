
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, datasets, tree
import graphviz 

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

print("So lop: ", len(np.unique(iris_y)))
print("So du lieu kiem thu: ", len(iris_y))


X0 = iris_X[iris_y == 0, :]
print("Mau don gian tu lop 0: \n", X0[:5, :])

X1 = iris_X[iris_y == 1, :]
print("Mau don gian tu lop 1: \n", X1[:5, :])

X2 = iris_X[iris_y == 2, :]
print("Mau don gian tu lop 2: \n", X2[:5, :])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
     iris_X, iris_y, test_size=50)

print("tap huan luyen: ", len(y_train))
print("kich thuoc kiem thu: ", len(y_test))

clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris_X, iris_y)

dot_data = tree.export_graphviz(clf, 
	out_file=None,
	feature_names=iris.feature_names,  
	class_names=iris.target_names,  
	filled=True, 
	rounded=True,  
	special_characters=True
	)

# dot_data_nonsetfield = tree.export_graphviz(clf, 
# 	out_file=None
# 	)

graph = graphviz.Source(dot_data)
graph.render("iris_decision_tree")

# graph_nonfield = graphviz.Source(dot_data_nonsetfield)
# graph_nonfield.render("iris non field")