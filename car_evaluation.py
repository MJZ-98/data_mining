import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import classification_report

data = pd.read_csv("C:/Users/MJZ/Desktop/dataset/car.data",header=None) # 读取数据
name = pd.read_csv("C:/Users/MJZ/Desktop/dataset/car.c45-names",header=0,sep='\n')   #
print(data,data.info(),name)  #查看数据特征 属性名

names = []
for i in range(3,name.shape[0]):
    names.append(name.iloc[i,:].str.split(':')[0]   [0])
names += ['car_acceptability']
data.columns = names
print(data)
print(data['car_acceptability'].value_counts()/data.shape[0])
plt.figure()
data['car_acceptability'].value_counts().plot(kind='pie')
plt.show()
'''
# 查看数据集中汽车可接受情况的分布情况:
unacc    0.700231
acc      0.222222
good     0.039931
vgood    0.037616
'''

# --------------------------特征分析--------------------------------------
'''
汽车评价数据库是由一个简单的层次结构派生而来的决策模型，最初是为DEX的演示而开发的
（M. Bohanec, V. Rajkovic:决策专家系统制作。Sistemica 1(1)，页145-157,1990.)。
模型的评估汽车按照以下概念结构:
。价格总体价格
。购买购买价格
。维护费用
。科技技术特点
。舒适度
。门数
。以人来衡量人的承载能力
。lug_boot后备箱的大小
。估计汽车的安全性

除了目标，概念(CAR)，模型包括三个中间概念:价格、科技、舒适。每个概念都在原始模型中
由一组实例关联到它的较低级别的后代汽车评价数据库包含了结构型的实例信息删除,
即直接关系到汽车的六个输入属性:购买费，维护费，车门数，载人数，后备箱空间，安全性。
'''

# --------------------------处理非数值特征--------------------------------------
"""
buying: 将超高价设为4，高价设为3，中价设为2，低价设为1
maint: 将超高价设为4，高价设为3，中价设为2，低价设为1
doors: 将'5more'设为5，处理后的数据为[2,3,4,5]
persons: 将'more'设为6，处理后的数据为[2,4,6]
lug_boot: 将'small'设为1, 'med'设为2, 'big'设为3
safety: 将'low'设为1, 'med'设为2, 'hjgh'设为3
car_acceptability: 将'unacc'设为1, 'acc'设为2, 'good'设为3, 'vgood'设为4
"""
def deal_data(data):
    data['buying'].replace(['vhigh', 'high','med','low'], [1,2,3,4], inplace=True)
    data['maint'].replace(['vhigh', 'high','med','low'],  [1,2,3,4], inplace=True)
    data['doors'].replace(['2','3','4','5more'], [2,3,4,5], inplace=True)
    data['persons'].replace(['2','4','more'], [2,4,6], inplace=True)
    data['lug_boot'].replace(['small','med','big'], [1,2,3], inplace=True)
    data['safety'].replace(['low','med','high'], [1,2,3], inplace=True)
    data['car_acceptability'].replace(['unacc','acc','good','vgood'], [1,2,3,4], inplace=True)

deal_data(data)
print(data)
pd.set_option('display.max_columns',None)
print(data.corr())

x = data.iloc[:,:-1].values
y = data.iloc[:,-1]
DT = tree.DecisionTreeClassifier(criterion="gini",splitter='best',\
                                 max_depth=11,min_samples_leaf=1,min_samples_split=2)
# #------------------------------------------------------
# # X_train, X_test, y_train, y_test = train_test_split(x,y , test_size=0.1,random_state=666)
# # mlp.fit(X_train,y_train)
# # y_predict = mlp.predict(X_test)
# # print('-score:',mlp.score(X_test,y_test))
# # print(classification_report(y_predict,y_test))
# #-------------------------------------------------
from sklearn.model_selection import KFold
kf = KFold(n_splits=10, shuffle=True)
i = 0
mean_train = []
mean_test = []
for train_index, test_index in kf.split(x):
    i += 1
    X_train, X_test = x[train_index], x[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("\n第%d折:"%i)
    DT.fit(X_train, y_train)
    y_pred_test = DT.predict(X_test)
    mean_train.append(DT.score(X_train, y_train))
    mean_test.append(DT.score(X_test, y_test))
    print( "train_score:", DT.score(X_train, y_train), "test score:", DT.score(X_test, y_test))
    print(classification_report(y_pred_test, y_test))
print('the mean of accracy of train:',sum(mean_train)/len(mean_train),\
      '\nthe mean of accracy of test',sum(mean_test)/len(mean_test))
