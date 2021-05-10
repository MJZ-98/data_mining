import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

def rfc(X_train,X_test,y_train,y_test):
    clf = RandomForestClassifier(n_estimators=30,criterion='entropy',oob_score=True)
    clf.fit(X_train,y_train)
    Y_pred = clf.predict(X_test)
    print('-score:',clf.score(X_test,y_test))
    print('-oob score:',clf.oob_score_)
    savetress(clf)
    return Y_pred

def savetress(clf):
    pass

wine = pd.read_csv("C:/Users/MJZ/Desktop/dataset/winedata.csv")
X = wine.iloc[:,1:]
# X = (X-X.min())/(X.max()-X.min())
y = wine.iloc[:,0]
X=X.astype(float)
y=y.astype(int)
X=X.values

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=50, test_size=0.2)
y_pre=rfc(X_train, X_test, y_train, y_test)
