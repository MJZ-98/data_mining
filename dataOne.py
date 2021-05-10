from sklearn import preprocessing, datasets
from sklearn.decomposition import PCA
import numpy as np
import pdd as pd
import matplotlib.pyplot as plt

# 属性规约
boston = datasets.load_boston()
X = preprocessing.scale(boston.data)
pca = PCA(n_components='mle')
pca.fit(X)
print(sum(pca.explained_variance_ratio_))

# 数值规约
# data = np.random.randint(1,10,10)
# print(data)
# bins = np.linspace(data.min(),data.max(),4,endpoint = True)
# plt.hist(data,bins=bins,rwidth=0.95,edgecolor='k')
# plt.show()
#           随机抽样
# iris = datasets.load_iris()
# iris_df = pd.DataFrame(iris.data)
# iris_df.columns = iris.feature_names
# print(iris_df.sample(n = 10,replace=True))
# print(iris_df.sample(frac = 0.5))
#           分层抽样
# iris_df['target'] = iris.target
# A=iris_df[iris_df.target==0].sample(frac=0.2)
# B=iris_df[iris_df.target==1].sample(frac=0.3)
# print(A.append(B))