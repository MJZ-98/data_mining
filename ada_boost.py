import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import  *
from sklearn import tree
import graphviz
import numpy as np
from sklearn.model_selection import cross_val_score

X=[(1.60, 66),    (1.59, 60), (1.62, 59), (1.78, 80), (1.68, 70), (1.79, 59), (1.53, 40),
   (1.63,65), (1.65,50), (1.85,69),(1.55,90)]
y=[1, -1, 1, -1, 1, -1, 1, -1, 1, -1 , -1] #1女 -1 男
d=np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1,0.1])

m = 11
clf = tree.DecisionTreeClassifier(max_depth=3)
clf.fit(X,y,sample_weight=d)
y_predict = clf.predict(X)
e1 = 0
for i in range(m):
    if (y[i]!=y_predict[i]):
        e1 += d[i]
print("erro:",e1)
alpha = float(0.5*np.log((1.0 - e1)/max(e1,1e-16)))
print("a1:",alpha)
total = 0
for i in range(m):
    if (y[i]==y_predict[i]):
        d[i] = d[i]*np.exp(-alpha)
    else:
        d[i] = d[i]*np.exp(alpha)
    total += d[i]
print("D:",d)
d = d/total
print("pred:",y_predict)

lf = tree.DecisionTreeClassifier(criterion="gini",max_depth=3)
clf.fit(X,y,sample_weight=d)
y_predict = clf.predict(X)
e1 = 0
for i in range(m):
    if (y[i]!=y_predict[i]):
        e1 += d[i]
print("erro:",e1)
alpha = float(0.5*np.log((1.0 - e1)/max(e1,1e-16)))
print("a2:",alpha)
total = 0
for i in range(m):
    if (y[i]==y_predict[i]):
        d[i] = d[i]*np.exp(-alpha)
    else:
        d[i] = d[i]*np.exp(alpha)
    total += d[i]
print("D:",d)
d = d/total

print("pred:",y_predict)

for clf_name, clf in models:
    rfc_s = cross_val_score(clf,X,y,cv=10,scoring='accuracy')
    print(rfc_s)
    clf.fit(X_train, y_train)
    print(clf_name,"train score:",clf.score(X_train,y_train),'test score:',clf.score(X_test,y_test))
    y_pred_test = clf.predict(X_test)
    print(clf_name, "train_score:", clf.score(X_train, y_train), "test score:", clf.score(X_test, y_test))