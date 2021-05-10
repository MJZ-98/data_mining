import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
X=np.array([[34.62366,78.02469],
[30.28671,43.895],
[35.84741,72.9022],
[60.1826,86.30855],
[79.03274,75.34438],
 [45.08328,56.31637],
[61.10666,96.51143],
[75.02475,46.55401],
[76.09879,87.42057],
[84.43282,43.53339]])
y=np.array([0,0,0,1,1,0,1,1,1,1])
X0 = (X - X.mean())/X.std()

from sklearn.linear_model import LogisticRegression

# SVM.svc分类算法
from sklearn.svm import  *
model = SVC(kernel='linear',C=1e20)
model.fit(X0,y)
print(model.coef_)
print(model.intercept_)
print("SVC score:",model.score(X,y))
x1=np.linspace(X0.min(),X0.max(), 100)
x2=(-model.coef_[0,0]*x1-model.intercept_)/model.coef_[0,1]

plt.plot(x1, x2)

l = LogisticRegression(C=1.0, solver = "liblinear")
l.fit(X0, y)
df=pd.DataFrame (X0, columns=["x1","x2"])
df['target']=y
x1=np.linspace(X0.min(),X0.max(), 100)
x2=(-l.coef_[0,0]*x1-l.intercept_)/l.coef_[0,1]
plt.scatter(df.loc[df["target"]==0]["x1"],df.loc[df["target"]==0]["x2"],marker = '*' ,color ='r')
plt.scatter(df.loc[df["target"]==1]["x1"],df.loc[df["target"]==1]["x2"],marker = 'o' ,color ='g')
plt.plot(x1, x2)
plt.legend(['Linear','SVM'])
plt.show()
print("sklearns 参数:")
print(l.coef_ )
print(l.intercept_ )

X_test = np.array([[40,25],[60,100]])
X_test = (X_test - X.mean())/X.std()
print("X_test after Z-score:",'\n',X_test)
print("The predict results:",'\n',l.predict(X_test))
print("The score:",'\n',l.score(X0, y))
Y_pred = l.predict(X0)

from sklearn.metrics import accuracy_score, recall_score, f1_score
print(" -ACC:", accuracy_score(y, Y_pred))
print(" -REC:",recall_score(y, Y_pred))
print(" -F1: ",f1_score(y, Y_pred))