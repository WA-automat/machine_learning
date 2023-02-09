import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.model_selection import GridSearchCV, cross_val_score  # 网格搜索，交叉验证
from sklearn.tree import DecisionTreeClassifier
import graphviz

sns.set()

print(os.getcwd())
df = pd.read_csv('heart.csv')

# 基本信息
# print(df.head())
# print(df.info())

# 切分训练集与测试集
x = df.iloc[:, df.columns != "target"]
y = df.iloc[:, df.columns == "target"]
# print(x)
# print(y)
Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.3, random_state=40)
for i in [Xtrain, Xtest, Ytrain, Ytest]:
    i.index = range(i.shape[0])

# 查找最合适的最大树深
tr = []
te = []
for i in range(10):
    clf = DecisionTreeClassifier(random_state=25, max_depth=i + 1, criterion="entropy")
    clf = clf.fit(Xtrain, Ytrain)
    score_tr = clf.score(Xtrain, Ytrain)
    score_te = cross_val_score(clf, x, y, cv=10).mean()
    tr.append(score_tr)
    te.append(score_te)
# print(max(te))
plt.figure(figsize=(9, 7), dpi=200, facecolor="#EFE9E6")
plt.plot(range(1, 11), tr, marker='o', label="train")
plt.plot(range(1, 11), te, marker='o', label="test")
plt.fill_between(x=range(1, 11), y1=tr, y2=te, alpha=0.5)

plt.xlabel('depth', fontsize=16)
plt.ylabel('accuracy', fontsize=16)
plt.title('Optimal tree depth', fontsize=20)
plt.legend()
plt.savefig('Optimal-tree-depth')
plt.show()

# 网格搜索
parameters = {
    "min_samples_leaf": [*range(1, 20)],
    "min_impurity_decrease": [*np.linspace(0, 0.01, 1)]
}

clf = DecisionTreeClassifier(random_state=25,
                             max_depth=3,
                             criterion="entropy",
                             min_samples_leaf=11,
                             min_impurity_decrease=0.0)
GS = GridSearchCV(clf, parameters, cv=10)
GS = GS.fit(Xtrain, Ytrain)

# print(GS.best_params_)
# print(GS.best_score_)

# 决策树模型的训练
clf = DecisionTreeClassifier(random_state=25,
                             max_depth=3,
                             criterion="entropy",
                             min_samples_leaf=11,
                             min_impurity_decrease=0.0)
clf.fit(Xtrain, Ytrain)

# 综合得分
score = clf.score(Xtest, Ytest)
print('综合得分 =', score)

# 交叉验证
score = cross_val_score(clf, x, y, cv=10).mean()
print('交叉验证得分 =', score)

# 各属性值权重
importance = clf.feature_importances_
feature = pd.Series(importance, index=Xtrain.columns)
feature.sort_values(inplace=True, ascending=False)
# print(feature)

# 绘制权重柱状图
plt.figure(figsize=(9, 7), dpi=200, facecolor="#EFE9E6")
plt.bar(feature.index, height=feature)
plt.title('feature weight', fontsize=20)
plt.xlabel('feature', fontsize=16)
plt.ylabel('weight', fontsize=16)
plt.xticks(rotation=30)
plt.savefig('feature-weight')
plt.show()

dot_data = tree.export_graphviz(clf, out_file=None,
                                feature_names=Xtrain.columns,
                                filled=True,
                                rounded=True)
graph = graphviz.Source(dot_data)
graph.render('heart')
graph.view()

# 程序结束
print('Done!')
