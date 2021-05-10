import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def show(Y_pred):
    x = range(X_test.shape[0])
    fig = plt. figure()
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.set_title("Predict class")
    ax2.set_title('True class')
    ax1.scatter(x, Y_pred, c=Y_pred, marker='o' )
    ax2.scatter(x, y_test, c=y_test, marker='s' )
    plt. show()


def knn(X_train, X_test, y_train, y_test):
    clf = KNeighborsClassifier(n_neighbors=5) #algori thmhlli KDtree
    clf.fit(X_train, y_train)
    Y_pred = clf.predict(X_test)
    print("--score:", clf.score(X_test,y_test))
    return Y_pred
#
# # iris=datasets.load_iris()
# # X=iris.data
# # y=iris.target
# # X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0, test_size=0.2)
# # y_pre=knn(X_train, X_test, y_train, y_test)
# # show(y_pre)
#
wine = pd.read_csv("C:/Users/MJZ/Desktop/dataset/winedata.csv")
X = wine.iloc[:,1:]
X = (X-X.min())/(X.max()-X.min())
y = wine.iloc[:,0]
X=X.astype(float)
y=y.astype(int)
X=X.values

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=50, test_size=0.2)
y_pre=knn(X_train, X_test, y_train, y_test)
show(y_pre)

from sklearn.model_selection import KFold
kf = KFold(n_splits=2, shuffle=True)
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("kFold/: %s %s" % (X_train.shape, X_test.shape))
    y_pre=knn(X_train, X_test, y_train, y_test)
    show(y_pre)


