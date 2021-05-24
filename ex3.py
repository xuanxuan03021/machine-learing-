## Machine Learning Online Class - Exercise 3 | Part 1: One-vs-all
import matplotlib.pyplot as plt
import scipy.io
import numpy as np
from matplotlib import use
use('TkAgg')
import math
import scipy.optimize as op

np.set_printoptions(threshold = 1e6)#设置打印数量的阈值


#  Instructions
#  ------------
# 
#  This file contains code that helps you get started on the
#  linear exercise. You will need to complete the following functions 
#  in this exericse:
#
#     lrCostFunction.m (logistic regression cost function)
#     oneVsAll.m
#     predictOneVsAll.m
#     predict.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## Setup the parameters you will use for this part of the exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
num_labels = 10          # 10 labels, from 1 to 10
                         # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#
def displayData(x):
    width = round(math.sqrt(np.size(x, 1)))
    m, n = np.shape(x)
    print(m,n)
    height = int(n/width)
    # 显示图像的数量
    drows = math.floor(math.sqrt(m))
    print(drows)
    dcols = math.ceil(m/drows)

    pad = 1
    # 建立一个空白“背景布”
    darray = -1*np.ones((pad+drows*(height+pad), pad+dcols*(width+pad)))

    curr_ex = 0
    for j in range(drows):
        for i in range(dcols):
            if curr_ex >= m:
                break
            max_val = np.max(np.abs(X[curr_ex, :]))
            darray[pad+j*(height+pad):pad+j*(height+pad)+height, pad+i*(width+pad):pad+i*(width+pad)+width]\
                = x[curr_ex, :].reshape((height, width))/max_val
            curr_ex += 1
        if curr_ex >= m:
            break

    plt.imshow(darray.T, cmap='gray')
    plt.show()

# Load Training Data
print('Loading and Visualizing Data ...')

data = scipy.io.loadmat('ex3data1.mat') # training data stored in arrays X, y
X = data['X']
y = data['y'][:,0]
print(y)
m, _ = X.shape

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]

displayData(sel)
_=input("Program paused. Press Enter to continue...")

## ============ Part 2: Vectorize Logistic Regression ============
#  In this part of the exercise, you will reuse your logistic regression
#  code from the last exercise. You task here is to make sure that your
#  regularized logistic regression implementation is vectorized. After
#  that, you will implement one-vs-all classification for the handwritten
#  digit dataset.
#

print ('Training One-vs-All Logistic Regression...')

def sigmoid(x):
    sig=1/(1+np.exp(-1*x))
    return sig

def costFunction(theta,x,y,lam):
    m=len(y)
    h_of_x=sigmoid(x.dot(theta))#
    J=(((-1*y).T).dot(np.log(h_of_x))-((1-y).T).dot(np.log(1-h_of_x)))/m+lam/(2*m)*theta[1:,].dot(theta[1:,].T)
    return J

def gradFuncReg(theta, x, y, lam):
    m = np.size(y, 0)
    h = sigmoid(x.dot(theta))
    grad = np.zeros(np.size(theta, 0))
    grad[0] = 1/m*(x[:, 0].dot(h-y))
    grad[1:] = 1/m*(x[:, 1:].T.dot(h-y))+lam*theta[1:]/m
    return grad

def oneVsAll(X, y, num_labels, Lambda):
    k=num_labels
    m,n= X.shape
    theta=np.zeros((10,n+1))
    X=np.concatenate((np.ones((m, 1)), X), axis=1)
    for i in range(k):
        if i==0:
            num=10
        else:
            num=i

        Y = (y == num)*1#一维数组不能直接穿数组进去，查找满足条件的，直接用y和其他数组比较，返回T/F
        initial_theta=np.zeros((np.size(X,axis=1),))
        result = op.minimize(costFunction, x0=initial_theta, method='BFGS', jac=gradFuncReg, args=(X, Y, Lambda))
        theta_cat= result.x
        theta[i,:]=theta_cat
    return theta





Lambda = 0.1
all_theta = oneVsAll(X, y, num_labels, Lambda)

_=("Program paused. Press Enter to continue...")


# ## ================ Part 3: Predict for One-Vs-All ================
#  After ...
def  predictOneVsAll (all_theta, X):
    m,n= X.shape
    X=np.concatenate((np.ones((m, 1)), X), axis=1)
    predict=X.dot(all_theta.T)
    cat=np.argmax(predict,axis=1)
    return cat


pred = predictOneVsAll(all_theta, X)
print('Training Set Accuracy: ', np.sum(pred == (y% 10))/np.size(y, 0))
