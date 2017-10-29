# -*- coding:utf-8 -*-

from LinearRegression import generate_train_and_test,LinearRegression,r2_score
from sklearn.linear_model import LinearRegression as L
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score as R
import matplotlib.pyplot as plt

"""数据导入——————————————————————————————————————"""

data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis = 1)
#归一化处理
X_train, X_test, y_train, y_test = generate_train_and_test(features.values, prices.values)
#print X_train.shape,y_train.shape

#end——————————————————————————————————————————
#print np.array(features)[:,2]
c=LinearRegression()
clf=L()
clf.fit(X_train,y_train)
loss=c.fit(X_train,y_train)
s=clf.predict(X_test)
S=c.predict(X_test)
print R(y_test,S)
print R(y_test,s)
plt.plot(loss,'r')
plt.show()
