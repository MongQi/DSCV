# -*- coding:utf-8 -*-
import numpy as np

LEN, seed = 489, None
def generate_train_and_test(X, y):
    """打乱并分割数据为训练集和测试集"""
    # 按照 100 和 389 来分配训练集和数据集
    # 100 / 489 = 0.204  接近于0.8：0.2的比例
    # random.seed():用于指定随机数生成时所用算法开始的整数值,如果使用相同的seed()值，则每次生成的随即数都相同
    # 如果不设置这个值，则系统根据时间来自己选择这个值,此时每次生成的随机数因时间差异而不同。
    """
    X[:, 0] = (X[:, 0] - np.mean(X[:, 0])) / (np.std(X[:, 0]))
    X[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / (np.std(X[:, 1]))
    X[:, 2] = (X[:, 2] - np.mean(X[:, 2])) / (np.std(X[:, 2]))
    y = (y - np.mean(y)) / np.std(y)
    """
    X = np.insert(X, 0, values=np.ones(LEN), axis=1)
    sequence = []
    # sequence = np.array([], dtype=np.int64)
    np.random.seed(1997)
    while (True):
        r = np.random.randint(0, LEN)
        if (r not in sequence):
            # sequence=np.append(sequence,r)
            sequence.append(r)
            # print (sequence)
        if (len(sequence) > 99): break
    test_arg = np.array(sequence, dtype=np.int32)
    test_arg.sort()
    train_arg = np.delete(np.arange(0, LEN, 1), test_arg)
    X_test = X[test_arg]
    X_train = X[train_arg]
    y_test = y[test_arg]
    y_train = y[train_arg]
    return X_train, X_test, y_train, y_test

def r2_score(self,y_true,y_test):
        """计算并返回预测值相比于预测值的分数"""
        # R²=1-残差平方和/总离差平方和
        y_true = np.array(y_true)
        y_test = np.array(y_test)
        score = 1 - np.sum(np.square(y_true - y_test)) / (np.sum(np.square(y_true - np.mean(y_true))))
        return score

class LinearRegression(object):
    #线性回归 回归函数————
    # #由于有三个影响房价的因素
    # >>>>>f(x)= a0 + a1 * X1 + a2 * X2 + a3 * X3
    def __init__(self):
        self.theta=np.array([[0.],[1.],[1.],[1.]])
        self.alpha = 1
    def f_(self,X):
        return np.dot(X,self.theta)


    def loss(self,X,y):
        loss= np.sum((self.f_(X)-y)**2)/(2*len(y))
        return loss
    def show(self,X,y):
        self.X_train=X
        print self.f_(X)
        print self.loss(X,y)
    def fit(self,X_train,y_train):
        # f(x)= a0 + a1 * X1 + a2 * X2 + a3 * X3
        # J(A)= ( ∑ (f(Xi)-Yi)² ) / 2m
        # ∂ J(A) / a0 = a0 - α* ∑ ( f(Xi)-Yi)* 1 ) / m
        # ∂ J(A) / a1 = a1 - α* ∑ ( f(Xi)-Yi)*X1i) / m
        # ∂ J(A) / a2 = a2 - α* ∑ ( f(Xi)-Yi)*X2i) / m
        # ∂ J(A) / a3 = a3 - α* ∑ ( f(Xi)-Yi)*X3i) / m
        # x:['RM'，       'LSTAT'，                  'PTRATIO']
        # 住宅平均房间数量   区域中被认为是低收入阶层的比率 镇上学生与教师数量比例
        self.X_train_mean=[np.mean(X_train[:, 1]),np.mean(X_train[:, 2]),np.mean(X_train[:, 3])]
        self.X_train_std=[np.std(X_train[:, 1]),np.std(X_train[:, 2]),np.std(X_train[:, 3])]
        self.y_train_mean= np.mean(y_train)
        self.y_train_std =  np.std(y_train)
        X_train[:, 1] = (X_train[:, 1] - np.mean(X_train[:, 1])) / (np.std(X_train[:, 1]))
        X_train[:, 2] = (X_train[:, 2] - np.mean(X_train[:, 2])) / (np.std(X_train[:, 2]))
        X_train[:, 3] = (X_train[:, 3] - np.mean(X_train[:, 3])) / (np.std(X_train[:, 3]))
        y_train = (y_train - np.mean(y_train)) / np.std(y_train)
        loss=[]
        for i in range(100):
            #print self.loss(X_train,y_train)
            #print X_train[:,0]
            Variety_theta0=np.dot(self.f_(X_train).reshape(len(y_train))-y_train,X_train[:,0].T)/len(y_train)
            Variety_theta1=np.dot(self.f_(X_train).reshape(len(y_train))-y_train,X_train[:,1].T)/len(y_train)
            Variety_theta2=np.dot(self.f_(X_train).reshape(len(y_train))-y_train,X_train[:,2].T)/len(y_train)
            Variety_theta3=np.dot(self.f_(X_train).reshape(len(y_train))-y_train,X_train[:,3].T)/len(y_train)
            z=self.alpha*(np.array([Variety_theta0,Variety_theta1,Variety_theta2,Variety_theta3]).reshape(4,1))
            self.theta=self.theta-z
            loss.append(self.loss(X_train,y_train))
            #print z.shape,self.theta.shape
            print self.theta
            #print self.loss(X_train,y_train)
            #print self.theta
            #print Variety_theta0
        return loss
    def predict(self,X_test):
        X_test[:, 1] = (X_test[:, 1] - self.X_train_mean[0]) / (self.X_train_std[0])
        X_test[:, 2] = (X_test[:, 2] - self.X_train_mean[1]) / (self.X_train_std[1])
        X_test[:, 3] = (X_test[:, 3] - self.X_train_mean[2]) / (self.X_train_std[2])
        r=self.f_(X_test)
        r=(r*self.y_train_std+self.y_train_mean)
        return r

