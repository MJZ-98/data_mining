import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('C:/Users/MJZ/Desktop/something/machine learn/dataset/titanic_train.csv')  # 读取数据

# 绘图准备
fig = plt.figure(figsize=(16, 16), dpi=80)

print(data.columns)
pd.set_option('display.max_columns', None)  # 显示所有列
# print(data.describe())
print(data.info())
print('...................................')
print(data['Survived'].value_counts())  # 查看训练集中获救人数(342)和遇难人数(549)
print('...................................')

# --------------------------特征分析--------------------------------------
# 1. 考虑Pclass(船舱等级)各级的获救和遇难情况
print(data.groupby(['Pclass', 'Survived'])['Survived'].count())
print('...................................')
"""
运行结果：

Pclass  Survived
1       0            80
        1           136
2       0            97
        1            87
3       0           372
        1           119
Name: Survived, dtype: int64

经过groupby聚合，观察统计的数据之后，可以发现
1等船舱的总人数为216,获救率为62.9%;
2等船舱的总人数为184,获救率为47.2%;
3等船舱的总人数为491,获救率为24.2%.
说明船舱的等级与获救应该是有一定的关联，这个特征需要保留分析
"""

# 2. 特征Name中的数据特别复杂，而且根据经验判断乘客的姓名应该是和获救
# 没有什么关系的，因此不考虑特征Name


# 3. 分析特征Sex
print(data.groupby(['Sex', 'Survived'])['Survived'].count())
# 由于使用0和1表示是否获救，因此可以先将男女分开来看，计算各自的均值
# 就可以知道男性和女性各自的获救概率(使用bar柱状图能够直观比较各自的概率)
ax1 = fig.add_subplot(221)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['famale', 'male'])
ax1.bar([0, 1], data[['Survived', 'Sex']].groupby('Sex').mean()['Survived'])
print('...................................')
"""
运行结果:

Sex     Survived
female  0            81
        1           233
male    0           468
        1           109
Name: Survived, dtype: int64
从数据和可视化柱状图明显能够看出female(女性)的获救概率要远高于男性,
所以Sex也是我们必须要参考和选择的特征.
"""

# 4. 分析特征Age
"""
通过data.describe(),可以发现Age特征是有缺失值的.891条数据,
Age特征中只有714条数据,说明缺失了177条数据。
1. 省略缺失值
2. 考虑使用均值,中值,众数填充
3. 使用回归的方式进行缺失填充
"""
# 1. 省略
# print(data.dropna())
# 2. 填充:名字特征中有类似于Mr.(先生), Miss.(女士), Master.(少爷)的前缀,
#         可以猜测年龄与这些前缀有关,将前缀设置为新的特征,用于后续
#         计算每个前缀对应年龄的均值,对缺失值进行填充
data['Named'] = 0  # 创建新特征
for i in data:
    # 正则表达式,[A-Za-z]+表示多个大小写字母,以.作为结束的标志
    data['Named'] = data['Name'].str.extract('([A-Za-z]+)\.')

# print(data.groupby(['Sex', 'Named'])['Named'].count())

"""
Sex     Named   
female  Countess(伯爵夫人)      1
        Dr(博士)                1
        Lady(女士)              1
        Miss(小姐)             182
        Mlle((法语)小姐)        2
        Mme(夫人)               1
        Mrs(太太)              125
        Ms(女士)                1
male    Capt(船长)              1
        Col(上校)               2
        Don(阁下)               1
        Dr(博士)                6
        Jonkheer(洋奇家族)      1
        Major(陆军上校)         2
        Master(少爷)           40
        Mr(先生)              517
        Rev(牧师)              6
        Sir(先生)              1
Name: Named, dtype: int64
前缀称呼特别多,所以将人数比较少的归为人数多的类别,将可能年龄相近的归为一类,
有争议不清楚的称呼，就归为'Other'类
"""
data['Named'].replace(
    ['Countess', 'Dr', 'Lady', 'Mlle', 'Mme', 'Ms', 'Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir'],
    ['Mrs', 'Mr', 'Mrs', 'Miss', 'Miss', 'Miss', 'Mr', 'Other', 'Mr', 'Other', 'Mr', 'Other', 'Mr'], inplace=True
)
# print(data['Named'])
# 年龄为具体的数字,使用均值填充
print(data.groupby('Named')['Age'].mean())
print('...................................')

"""
Named
Master     4.574167
Miss      21.860000
Mr        32.739609
Mrs       35.981818
Other     45.888889
Name: Age, dtype: float64
用以上的均值对缺失值进行填充
"""
data.loc[(data.Named == 'Master') & (data.Age.isnull()), 'Age'] = 4.5
data.loc[(data.Named == 'Miss') & (data.Age.isnull()), 'Age'] = 21.8
data.loc[(data.Named == 'Mr') & (data.Age.isnull()), 'Age'] = 32.7
data.loc[(data.Named == 'Mrs') & (data.Age.isnull()), 'Age'] = 35.9
data.loc[(data.Named == 'Other') & (data.Age.isnull()), 'Age'] = 45.8

# print(data.describe()['Age'])  # 验证是否填充完整
# 分析各年龄层级获救的情况: 计算均值就是该年龄层的获救比例
print(data[['Named', 'Survived']].groupby('Named').mean())
print('...................................')
"""
        Survived
Named           
Master  0.575000
Miss    0.704301
Mr      0.162571
Mrs     0.795276
Other   0.111111
可以看出Master、Miss、Mrs的获救比例更大，说明妇女和儿童的获救比例更大
因此Age是我们需要关心的特征
"""

# 5. 一起分析特征SibSp(乘客的兄弟姐妹数量)和Parch(父母和孩子的数量)
print(data[['SibSp', 'Survived']].groupby('SibSp').mean())
print(data[['Parch', 'Survived']].groupby('Parch').mean())
print('...................................')
"""
       Survived
SibSp          
0      0.345395
1      0.535885
2      0.464286
3      0.250000
4      0.166667
5      0.000000
8      0.000000
       Survived
Parch          
0      0.343658
1      0.550847
2      0.500000
3      0.600000
4      0.000000
5      0.200000
6      0.000000
都有共同特点，就是如果兄弟姐妹很多或者父母和孩子的值比较大，获救的比例都很小
这两个特征也需要保留
"""

# 6. 分析特征Fare(船票的价格)
#    使用直方图进行大致的观察(hist)
ax2 = fig.add_subplot(222)
b = 50  # 组数
num_bins = (data['Fare'].max() - data['Fare'].min()) // b
ax2.hist(data['Fare'].to_list(), int(num_bins))

# 7. 分析特征Cabin(船舱):只有204个乘客不是空值，因此这个特征可以不考虑

# 8. 分析特征Embarked(登船地点):
# 有两个缺失值, 由于缺失值相对比较少，可以直接忽略, 也可以用众数进行填充
data['Embarked'].fillna('S', inplace=True)
print(data.groupby(['Embarked', 'Pclass', 'Survived'])['Survived'].count())
print('...................................')
"""
Embarked  Pclass  Survived
C         1       0            26
                  1            59
          2       0             8
                  1             9
          3       0            41
                  1            25
Q         1       0             1
                  1             1
          2       0             1
                  1             2
          3       0            45
                  1            27
S         1       0            53
                  1            74
          2       0            88
                  1            76
          3       0           286
                  1            67
Name: Survived, dtype: int64
经过简单分析可以得到的结论：
1. S地点上船的人数最多
2. Q地点上船的几乎都是三等船舱的人(Q地点上船的乘客几乎都很贫穷)
3. C地点的乘客获救比例为55.3%,Q地点的乘客获救比例为38.9%,S地点的乘客获救比例为33.6%
特征Embarked需要保留
"""

# --------------------------数据清洗--------------------------------------
# 1. 处理特征Age, 对连续值进行离散化
data['Age_mark'] = 0  # 为data创建新特征
# Age的最大值为80, 最小值为0.42, 将Age分成4组, 80/4=20
data.loc[data['Age'] <= 20, 'Age_mark'] = 0
data.loc[(data['Age'] > 20) & (data['Age'] <= 40), 'Age_mark'] = 1
data.loc[(data['Age'] > 40) & (data['Age'] <= 60), 'Age_mark'] = 2
data.loc[(data['Age'] > 60) & (data['Age'] <= 80), 'Age_mark'] = 3
# print(data.groupby(['Age_mark'])['Age_mark'].count())  # (20, 40]年龄段的人数最多

# 2. 处理SibSp和Parch  都是与家庭有关，可以合并为一个特征:家庭成员数量Family
data['Family'] = 0  # 为data创建新特征
data['Family'] = data['SibSp'] + data['Parch']

# 3. 处理特征Fare, 对连续值进行离散化
data['Fare_mark'] = 0  # 为data创建新特征
"""
# 最大值为512.3292, 最小值为0, 分成4组
# sep = (data['Fare'].max() - data['Fare'].min()) / 4
# data.loc[data['Fare'] <= sep, 'Fare_mark'] = 0
# data.loc[(data['Fare'] > sep) & (data['Fare'] <= (2*sep)), 'Fare_mark'] = 1
# data.loc[(data['Fare'] > (2*sep)) & (data['Fare'] <= (3*sep)), 'Fare_mark'] = 2
# data.loc[(data['Fare'] > (3*sep)) & (data['Fare'] <= (4*sep)), 'Fare_mark'] = 3
# print(data[['Fare_mark', 'Survived']].groupby('Fare_mark').mean())
"""
# 分箱(等频),  平均划分船票价格效果不理想, 为了更好地平滑噪声，对于船票使用之前所学的等频分箱
data['Fare_mark'] = pd.qcut(data['Fare'], 4)
# print(data.groupby(['Fare_mark'])['Survived'].mean())
"""
Fare_mark
(-0.001, 7.91]     0.197309
(7.91, 14.454]     0.303571
(14.454, 31.0]     0.454955
(31.0, 512.329]    0.581081
Name: Survived, dtype: float64
根据这个数据,可以看出船票价格越高,获救的概率更大
"""
data['Fare_marked'] = 0
data.loc[data['Fare'] <= 7.91, 'Fare_marked'] = 0
data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare_marked'] = 1
data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare_marked'] = 2
data.loc[(data['Fare'] > 31) & (data['Fare'] <= 513), 'Fare_marked'] = 3
print(data.groupby(['Fare_mark'])['Survived'].mean())
print('...................................')

# 4. 处理非数值特征
"""
Sex: 将女性设为0, 男性设为1
Embarked: 将C地点设为0, Q地点设为1, S地点设为2
Named: 将'Master'设为0, 'Miss'设为1, 'Mr'设为2, 'Mrs'设为3, 'Other'设为4
"""
data['Sex'].replace(['female', 'male'], [0, 1], inplace=True)
data['Embarked'].replace(['C', 'Q', 'S'], [0, 1, 2], inplace=True)
data['Named'].replace(['Master', 'Miss', 'Mr', 'Mrs', 'Other'], [0, 1, 2, 3, 4], inplace=True)

# 5. 去掉不必要的特征
data.drop(['PassengerId', 'Name', 'Age', 'Fare_mark', 'Fare', 'Cabin', 'Ticket'], axis=1, inplace=True)
# print(data.head(5))
# print(data.info())
"""
清洗之后的数据：
RangeIndex: 891 entries, 0 to 890
Data columns (total 10 columns):
Survived       891 non-null int64
Pclass         891 non-null int64
Sex            891 non-null int64
SibSp          891 non-null int64
Parch          891 non-null int64
Embarked       891 non-null int64
Named          891 non-null int64
Age_mark       891 non-null int64
Family         891 non-null int64
Fare_marked    891 non-null int64
dtypes: int64(10)
-------------------------------------
"""

# --------------------------建模:训练、预测--------------------------------------
X = data.iloc[:, 1:]
y = data.iloc[:, 0]
# 对比多种算法的acc
models = [
    ('KNN', KNeighborsClassifier(n_neighbors=3)),
    ('NBayes', GaussianNB()),
    ('ID3', DecisionTreeClassifier(criterion='entropy')),
    ('CART', DecisionTreeClassifier(criterion='gini')),
    ('SVM', SVC(kernel='linear', C=1e2)),
    ('Logistic', LogisticRegression(C=1.0, solver='liblinear', multi_class='ovr')),
    ('RF', RandomForestClassifier(n_estimators=20, criterion='entropy'))
]

# 1. 使用train_test_split划分数据集
print('train_test_split随机划分: ')
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.2)

for clf_name, clf in models:
    clf.fit(X_train, y_train)
    y_pred_test = clf.predict(X_test)
    print(clf_name, 'train score: ', clf.score(X_train, y_train),
          '  test score: ', clf.score(X_test, y_test))
print('-------------------------------------------')

# 2. 使用k折交叉验证
n = 8
print('%d折交叉验证: ' % n)
X, y = np.array(X), np.array(y)
kf = KFold(n_splits=n, random_state=10)
for clf_name, clf in models:
    train_score_tmp, test_score_tmp = 0, 0
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        y_pred_test = clf.predict(X_test)
        train_score_tmp += clf.score(X_train, y_train)
        test_score_tmp += clf.score(X_test, y_test)
    print(clf_name, 'train score: ', train_score_tmp / n,
          '  test score: ', test_score_tmp / n)  # 以平均值作为指标进行评估
print('-------------------------------------------')

# plt.show()
