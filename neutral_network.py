import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

wine = pd.read_csv("winedata.csv")
X = wine.iloc[:,1:]
X = (X-X.min())/(X.max()-X.min())
y = wine.iloc[:,0]
X=X.astype(float)
y=y.astype(int)
X=X.values

X_train, X_test, y_train, y_test = train_test_split(X,y , test_size=0.1)
mlp = MLPClassifier(solver='sgd',activation='identity',max_iter=500,hidden_layer_sizes=(100,50))
mlp.fit(X_train,y_train)
y_predict = mlp.predict(X_test)
print('-score:',mlp.score(X_test,y_test))
print(classification_report(y_predict,y_test))


