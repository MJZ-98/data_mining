import pdd as pd
import numpy as np
from sklearn import datasets
from sklearn import preprocessing
from sklearn.preprocessing import Binarizer

boston = datasets.load_boston()
print(boston.data,"\n",boston.data.shape,"\n",boston.feature_names)
print(boston.target,'\n',boston)

df = pd.DataFrame(boston.data[:,4:7])
df.columns = boston.feature_names[4:7]
print(df,'\n',df.info)

# 数据规范化
# df = (df-df.min())/(df.max()-df.min())        #最小最大规范化
# df = preprocessing.minmax_scale(df)
# df = (df-df.mean())/df.std()        #Z分数变化
# df = preprocessing.scale(df)
# df = df/10**np.ceil(np.log10(df.abs().max()))

# 连续属性离散化
# pd.cut(df.AGE,5,labels=range(5))
# print(pd.cut(df.AGE,5,labels=range(5)))
pd.qcut(df.AGE,5,labels=range(5))
print(pd.qcut(df.AGE[0:10],5,labels=range(5)))

# 特征二值化
# X = boston.target.reshape(-1,1)
# print(Binarizer(threshold=20.0).fit_transform(X))

# print(df)