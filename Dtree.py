# coding=gbk
from sklearn.metrics import classification_report
from sklearn import tree
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.datasets import load_iris
import graphviz

#pip install graphviz
#https://graphviz.gitlab.io/download/
#配置环境变量 把C:\Program Files (x86)\Graphviz2.38\bin 加入系统变量PATH

#5.4决策树
filename = 'C:/Users/MJZ/Desktop/dataset/winedata.csv'
data = pd.read_csv(filename)
print(data.columns)
y = data.iloc[:,0]
X = data.iloc[:,1:]
X = X.astype(float)
y = y.astype(int)


X_train, X_test, Y_train, Y_test = train_test_split(X, y, random_state=0)
#默认是gini
#dtc = tree.DecisionTreeClassifier()
dtc = tree.DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,Y_train)
y_predict = dtc.predict(X_test)

#获取结果报告
print('Accracy:',dtc.score(X_test,Y_test))
print(classification_report(y_predict,Y_test))
dot_data = tree.export_graphviz(dtc, out_file=None,
                         feature_names=data.columns[1:],
                         class_names=data.columns[0],
                         filled=True, rounded=True,
                         special_characters=True)

grz=graphviz.Source(dot_data)
grz.render("tree_wine")

