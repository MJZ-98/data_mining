import pdd as pd
import matplotlib.pyplot as plt

quotesdf_nan = pd.read_csv('C:/Users/MJZ/Desktop/1_missing.csv')    #导入数据集，创建索引
# print(quotesdf_nan.isnull())          #检查数据集中为空的数据，空值返回True
# print(quotesdf_nan.dropna(how='all'))         #删除行记录全为空值的数据
# print(quotesdf_nan.dropna())          #删除行记录有空值的数据
quotesdf_nan.fillna(quotesdf_nan.mean(),inplace=True)         #填充空值为样本平均值
# quotesdf_nan.fillna(method='bfill',inplace=True)          #填充空值为下条数据值
# print(quotesdf_nan.describe())          #查看数据整体情况
quotesdf_nan.loc[200]=[0.1,0.2,0.3,0.4]         #添加一条异常值数据
print(quotesdf_nan.iloc[:, 0:2])            #切片显示数据前两列
# quotesdf_nan.drop('d',axis=1).boxplot()           #查看'd'属性的箱型图
# plt.show()
print(quotesdf_nan)
# print(quotesdf_nan[abs(quotesdf_nan-quotesdf_nan.mean()) > 3*quotesdf_nan.std()].dropna(how='all'))   #自定义异常点排查

'''
今日作业：   1、观看3.3视频，并实现其中的代码。
            2、推导Z分数变换的均值和标准差。
            3、课前练习分箱平滑代码。
'''