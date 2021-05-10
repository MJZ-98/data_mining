from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import numpy as np

#12 24 33 2 4 55 68 26
df=pd.DataFrame({"data":[12,24,33,2,4,55,68,26]})
print(df.describe())

#加载数据
#a=np.random.randint(1,10,50)
iris=datasets.load_iris()
iris_df=pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df.columns=iris.feature_names
iris_df['target']=iris.target
#print(iris_df['target'])
iris_df['target']=iris.target.astype(float)

#直方图及正态分布检测
#print(iris_df.iloc[:,0])
plt.figure()
plt.subplot(121)
plt.hist(iris_df.iloc[:,0],30,color='c')
plt.subplot(122)
plt.hist(iris_df.iloc[:,1],30,color='c')
#plt.show()
#print(scipy.stats.normaltest(iris_df.iloc[:,0],axis=0))
stat,pvalue=scipy.stats.normaltest(iris_df.iloc[:,0],axis=0) #axis=0对数据每列进行正态分布检验
print(stat,pvalue)

stat,pvalue=scipy.stats.normaltest(iris_df.iloc[:,1],axis=0) #axis=0对数据每列进行正态分布检验
print(stat,pvalue)

#饼图
print(iris_df['target'].value_counts()) #统计每个目标分类的个数
plt.figure()
iris_df['target'].value_counts().plot(kind='pie')
#plt.show()

#统计量分析
print(iris_df.iloc[:,0].mean())
print(iris_df.iloc[:,0].median())
print(iris_df.iloc[:,0].std())
print(iris_df.iloc[:,0].quantile())
print(iris_df.iloc[:,0].quantile([0.25,0.75]))
iqr=iris_df.iloc[:,0].quantile([0.75]).loc[0.75]-iris_df.iloc[:,0].quantile([0.25]).loc[0.25]
print(iqr)
print(iris_df.iloc[:,0].describe())
iqr1=iris_df.iloc[:,0].describe().loc['75%']-iris_df.iloc[:,0].describe().loc['25%']
print(iqr1)

#相关性分析
#散点图 0列和2列
X1=iris.data[:,0]
X=[item[0] for item in iris.data]
Y1=iris.data[:,2]
Y=[item[2] for item in iris.data]

plt.figure()
plt.scatter(X[:50],Y[:50],color='red',marker = '*',label='1',s=10)#s为点的大小
plt.scatter(X[50:100],Y[50:100],color='green',marker = '.',label='2',s=20)
plt.scatter(X[100:150],Y[100:150],color='blue',marker = '+',label='3',s=30)
plt.xlabel("sepal length ")
plt.ylabel("petal length")
plt.legend(loc='best')
plt.show()

#相关系数 pearson
print(iris_df.iloc[:,[0,2,4]].corr())
print(iris_df['target'].corr(iris_df.iloc[:,0]))


"""
#课前练习参考代码1
X = [
    12.5, 15.3, 23.2, 26.4, 33.5,
    34.4, 39.4, 45.2, 55.4, 60.9
]
Y = [
    21.2, 23.9, 32.9, 34.1, 42.5,
    43.2, 49.0, 52.8, 59.4, 63.5
]
# 均值
XMean = np.mean(X);
YMean = np.mean(Y);
# 标准差
XSD = np.std(X);
YSD = np.std(Y);
# z分数
ZX = (X - XMean) / XSD;
ZY = (Y - YMean) / YSD;
# 相关系数
r = np.sum(ZX * ZY) / (len(X));
# 直接调用Python的内置的相关系数的计算方法
print(np.corrcoef(X, Y))

#课前练习参考代码2
X = [
    12.5, 15.3, 23.2, 26.4, 33.5,
    34.4, 39.4, 45.2, 55.4, 60.9
]
Y = [
    21.2, 23.9, 32.9, 34.1, 42.5,
    43.2, 49.0, 52.8, 59.4, 63.5
]

data = pd.DataFrame({
    'X': X,
    'Y': Y
})

print(data.corr())
"""