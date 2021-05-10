import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import  *
from sklearn.naive_bayes import GaussianNB
from sklearn import tree

import graphviz

wine = pd.read_csv("C:/Users/MJZ/Desktop/dataset/titanic_train.csv")
X = wine.iloc[:,1:]
# X = (X-X.min())/(X.max()-X.min())
y = wine.iloc[:,0]
X=X.astype(float)
y=y.astype(int)
X=X.values

# X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.1)

models = [
    ("KNN_3",KNeighborsClassifier(n_neighbors=3)),
    ("KNN_12",KNeighborsClassifier(n_neighbors=12)),
    ("Logistic",LogisticRegression(C=1.0, solver = "liblinear",multi_class="ovr")),
    ("bayes",GaussianNB()),
    ("DT_ID3",tree.DecisionTreeClassifier(criterion="entropy",min_samples_split=10,max_depth=3)),
    ("DT_gini",tree.DecisionTreeClassifier(criterion="gini",max_depth=3)),
    ("SVM",SVC(kernel='linear',C=1e20)),
    ("RF",RandomForestClassifier(n_estimators=300,criterion='gini',oob_score=True,max_depth=7,max_leaf_nodes=12,random_state=666)),
    ("MLP",MLPClassifier(solver='sgd',activation='identity',max_iter=500,hidden_layer_sizes=(100,50)))
]

from sklearn.model_selection import KFold
kf = KFold(n_splits=15, shuffle=True)
i = 0
mean_train = [[] for i in range(15)]
mean_test = [[] for i in range(15)]
for train_index, test_index in kf.split(X):
    i += 1
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("\n第%d折:"%i)
    for clf_name, clf in models:
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        mean_train[i-1].append(clf.score(X_train, y_train))
        mean_test[i-1].append(clf.score(X_test, y_test))
        print(clf_name, "train_score:", clf.score(X_train, y_train), "test score:", clf.score(X_test, y_test))

print('\nthe mean of train accuracy:')
for i in range(1):
    print(models[i][0],np.mean(mean_train,axis=0)[i])
print('\nthe mean of test accuracy:')
for i in range(1):
    print(models[i][0],np.mean(mean_test,axis=0)[i])
