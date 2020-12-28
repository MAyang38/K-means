import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import datasets
import pandas as pd
plt.rcParams['font.family'] = 'SimHei'

data = pd.read_csv('./iris_data.csv')
#表示我们只取特征空间中的后两个维度
X = data.drop(['sepal width','sepal length','target','label'],axis=1)
print(X.shape)
#绘制原始数据分布图;;
plt.scatter(X.values[:, 0], X.values[:, 1], c = "red", marker='o', label='see')
plt.title('数据原始分布')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()

#构造聚类器
estimator = KMeans(n_clusters=3, n_init=10,max_iter=300, tol=1e-4, precompute_distances='deprecated')
estimator.fit(X.values)#聚类
label_pred = estimator.labels_ #获取聚类标签
#绘制k-means结果
x0 = X[label_pred == 0]
x1 = X[label_pred == 1]
x2 = X[label_pred == 2]
plt.scatter(x0.values[:, 0], x0.values[:, 1], c = "red", marker='o', label='label0')
plt.scatter(x1.values[:, 0], x1.values[:, 1], c = "green", marker='*', label='label1')
plt.scatter(x2.values[:, 0], x2.values[:, 1], c = "blue", marker='+', label='label2')
plt.title('聚类分布')
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc=2)
plt.show()