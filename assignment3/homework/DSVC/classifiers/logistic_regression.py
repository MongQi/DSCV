
# -*- coding: utf-8 -*-
import numpy as np
import random
import math


class LogisticRegression(object):

    def __init__(self):
        self.w = None
        self.clf =None
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def loss(self, X_batch, y_batch):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
        data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """

        #########################################################################
        # TODO:                                                                 #
        # calculate the loss and the derivative                                 #
        #########################################################################
        #j(Θ)=Σ {y(i) log(hθ(x(i))) + (1−y(i)) log(1−hθ(x(i))) }/m
        z=X_batch.dot(self.w)
        loss=(-np.dot(y_batch.T,np.log(self.sigmoid(z)))-
              np.dot((1-y_batch.T),np.log(self.sigmoid(z))))/X_batch.shape[0]

        # ∂J(θ)/∂θj=∑(hθ(x(i))−y(i))*x(i)
        # ∂J(θ)/∂θ =∑(hθ(X_batch)-y_batch)*X_batch

        grad = np.dot(self.sigmoid(z)-y_batch,X_batch)/X_batch.shape[0]
        return loss,grad

        #########################################################################
        #                       END OF YOUR CODE                                #
        #########################################################################

    def train(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=True):

        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape

        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)

        loss_history = []

        for it in xrange(num_iters):
            arr=np.random.choice(xrange(num_train),batch_size,replace=False)
            X_batch = X[arr]
            y_batch = y[arr]
            """
            """
            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (batch_size, dim)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch)
            loss_history.append(loss)

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            self.w=self.w-grad*learning_rate

            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

        return loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: N x D array of training data. Each column is a D-dimensional point.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
        array of length N, and each element is an integer giving the predicted
        class.
        """
        y_pred = np.zeros(X.shape[1])
        y_pred = self.sigmoid(X.dot(self.w))
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        le=len(y_pred)
        i=0
        while(i<le):
            if(y_pred[i]>=0.5):y_pred[i]=1
            if(y_pred[i]<0.5):y_pred[i]=0
            i=i+1
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def one_vs_all(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose = True):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        """
        length_x=X.shape[0]
        y_test = np.ones([length_x, 10])
        for i in xrange(length_x):
            y_test[i][y[i]] = 0
        y_test=y_test.T
        self.clf=[]
        loss_history=np.zeros([10,num_iters])
        for i in range(10):
            self.clf.append(LogisticRegression())
        for j in range(10):
            print '关于数字 “%d” 的分类器的loss'% j
            loss_history[j]=self.clf[j].train(X,y_test[j],learning_rate,
                                         num_iters,batch_size,verbose)
        return loss_history


    def predict_one_vs_all(self, X):
        #用于对多分类器结果进行预测
        y_pred = np.zeros(X.shape[0])
        y_label=np.zeros([10,X.shape[0]])
        for i in range (10):
            y_label[i]=self.sigmoid(X.dot(self.clf[i].w))
        y_label=1-y_label
        y_label=y_label.T

        for i in range(y_label.shape[0]):
            y_pred[i]=np.argmax(y_label[i])
        return y_pred



    def train_with_l2(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose=True,Lambda=0.001):
        num_train, dim = X.shape

        if self.w is None:
            self.w = 0.001 * np.random.randn(dim)

        loss_history = []

        for it in xrange(num_iters):
            arr=np.random.choice(xrange(num_train),batch_size,replace=False)
            X_batch = X[arr]
            y_batch = y[arr]
            loss, grad = self.loss(X_batch, y_batch)
            #引入l2正则化后，代价函数实际变成
            #j(Θ)=Σ {y(i) log(hθ(x(i))) + (1−y(i)) log(1−hθ(x(i))) }/m+ λΣΘ /2m
            #不想更改loss函数，直接在train中更改loss和grad的值
            #λ暂时设置为0.001
            loss=loss-Lambda*np.sum(np.square(self.w))/(2*batch_size)
            loss_history.append(loss)
            self.w=self.w-grad*learning_rate-Lambda*np.sum(self.w)/batch_size
            if verbose and it % 200 == 0:
                print 'iteration %d / %d: loss %f' % (it, num_iters, loss)

        return loss_history


    def one_vs_all_with_l2(self, X, y, learning_rate=1e-3, num_iters=100,
            batch_size=200, verbose = True,Lambda=0.001):
        """
        Train this linear classifier using stochastic gradient descent.
        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
         training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels;
        - learning_rate: (float) learning rate for optimization.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        """
        length_x=X.shape[0]
        y_test = np.ones([length_x, 10])
        for i in xrange(length_x):
            y_test[i][y[i]] = 0
        y_test=y_test.T
        self.clf=[]

        loss_history=np.zeros([10,num_iters])
        for i in range(10):
            self.clf.append(LogisticRegression())
        for j in range(10):
            print '关于数字 “%d” 的分类器的loss'% j
            loss_history[j]=self.clf[j].train_with_l2(X,y_test[j],learning_rate,num_iters,batch_size,verbose,Lambda=Lambda)
        return loss_history