import pdd as pd
from sklearn import preprocessing
import  matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA

# x = [5,6,7,8,9,10,11,12,13]
# df = pd.DataFrame(x)
# # df = (df-df.min())/(df.max()-df.min())        #最小最大规范化
# df_minmax = preprocessing.minmax_scale(df)
# # df = (df-df.mean())/df.std()        #Z分数变化
# df_z = preprocessing.scale(df)
# 
# print(x,'\n',df_minmax,'\n',df_z)
# print(df_minmax.mean(),df_minmax.std(),"[",df_minmax.min(),df_minmax.max(),"]")
# print(df_z.mean(),df_z.std(),"[",df_z.min(),df_z.max(),"]")
# 
# fig = plt.figure(figsize=(16,8),dpi=70)
# ax = fig.add_subplot()
# ax.plot(range(len(x)),x,label='origin')
# ax.plot(range(len(x)),df_minmax,label='0-1')
# ax.plot(range(len(x)),df_z,label='Z')
# ax.legend(loc="best")
# plt.show()

# from sklearn import datasets
# import numpy as np
#
# iris = datasets.load_iris()
# R = np.array(iris.data)
# print(type(iris))
#
# R_cov = np.cov(R, rowvar=False)     #得到4*4协方差矩阵
# print(R,R_cov)
#
# import pandas as pd
# iris_covmat = pd.DataFrame(data=R_cov, columns=iris.feature_names)      #列标签
# iris_covmat.index = iris.feature_names      #行标签
#
# eig_values, eig_vectors = np.linalg.eig(R_cov)      #得到特征向量和特征值
#
# featureVector = eig_vectors[:,:2]       #切片特征向量值前两列
#
# featureVector_t = np.transpose(featureVector)
# R_t = np.transpose(R)
#
# newDataset_t = np.matmul(featureVector_t, R_t)
# newDataset = np.transpose(newDataset_t)
#
# plt.figure()
# idx_1 = np.where(iris.target==0)
# p1 =plt.scatter(newDataset[idx_1,0],newDataset[idx_1,1],marker = '*',color = 'r',label='1',s=10)
# idx_2 = np.where(iris.target==1)
# p2 = plt.scatter(newDataset[idx_2,0],newDataset[idx_2,1],marker = 'o',color ='g',label='2',s=20)
# idx_3 = np.where(iris.target==2)
# p3 = plt.scatter(newDataset[idx_3,0],newDataset[idx_3,1],marker = '+',color ='b',label='3',s=30)
#
# plt.legend((p1,p2,p3),(u'setosa',u'versicolor',u'virginica'),loc='best')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.show()



score = np.loadtxt("C:/Users/MJZ/Desktop/pca_test_score.csv",skiprows=1,delimiter=',')
R = np.array(score)
R_cov = np.corrcoef(R, rowvar=False)     #得到6*6相关系数矩阵

eig_values, eig_vectors = np.linalg.eig(R_cov)      #得到特征向量和特征值

print("特征向量：:",eig_vectors)
print("信息贡献度:",eig_values/6)
for i in range(1,len(eig_values)+1):
    if sum(eig_values[:i]/6)>0.9:
        print(eig_values[:i]/6,i)
        break
print(sum(eig_values))

# featureVector = eig_vectors[:,:3]       #切片特征向量值前两列
#
# featureVector_t = np.transpose(featureVector)
# R_t = np.transpose(R)
#
# newDataset_t = np.matmul(featureVector_t, R_t)
# newDataset = np.transpose(newDataset_t)
#
# print(featureVector_t)
#
# plt.figure()
# idx_1 = np.where(df['target']=="Y1")
# p1 =plt.scatter(newDataset[idx_1,0],newDataset[idx_1,1],marker = '*',color = 'r',label='1',s=10)
# idx_2 = np.where(df['target']=="Y2")
# p2 = plt.scatter(newDataset[idx_2,0],newDataset[idx_2,1],marker = 'o',color ='g',label='2',s=20)
# idx_3 = np.where(df['target']=="Y3")
# p3 = plt.scatter(newDataset[idx_3,0],newDataset[idx_3,1],marker = '+',color ='b',label='3',s=30)
#
# plt.legend((p1,p2,p3),(u'setosa',u'versicolor',u'virginica'),loc='best')
# plt.xlabel('PC1')
# plt.ylabel('PC2')
# plt.show()




# X = preprocessing.scale(score)
# pca = PCA(n_components=3)
# pca.fit(X)
# print(sum(pca.explained_variance_ratio_))