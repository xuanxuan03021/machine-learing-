## Machine Learning Online Class - Exercise 3 | Part 2: Neural Networks

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
from matplotlib import use
use('TkAgg')
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import math

np.set_printoptions(threshold = 1e6)#设置打印数量的阈值

# 显示随机100个图像, 疑问：最后的数组需要转置才会显示正的图像
def displayData(x):
    width = round(math.sqrt(np.size(x, 1)))
    m, n = np.shape(x)
    height = int(n/width)
    # 显示图像的数量
    drows = math.floor(math.sqrt(m))
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

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#
# LoadData
print('Loading and Visualizing Data ...')
datainfo = scipy.io.loadmat('ex3data1.mat')
X = datainfo['X']
Y = datainfo['y'][:, 0]
m = np.size(X, 0)
rand_indices = np.random.permutation(m)#将m个数打乱顺序
sel = X[rand_indices[0:100], :]
displayData(sel)
_ = input('Press [Enter] to continue.')
## ================ Part 2: Loading Pameters ================
# In this part of the exercise, we load some pre-initialized 
print('Loading Saved Neural Network Parameters ...')
weightinfo = scipy.io.loadmat('ex3weights.mat')
print(weightinfo)
theta1 = weightinfo['Theta1']
theta2 = weightinfo['Theta2']
#Theta1 has 25*401
#Theta2 has 26*10
# ================= Part 3: Implement Predict =================

# sigmoid函数
def sigmoid(z):
    g = 1/(1+np.exp(-1*z))
    return g

# 预测函数
def predict(theta1, theta2, x):
    m = np.size(x, 0)
    #x = np.concatenate((np.ones((m, 1)), x), axis=1)
    x = np.hstack((np.ones((m, 1)), x))  # 在X中加入拼接列,注意hstack和vstack 双括号
    temp1 = sigmoid(x.dot(theta1.T))#5000*25
    temp = np.concatenate((np.ones((m, 1)), temp1), axis=1)#5000*26
    temp2 = sigmoid(temp.dot(theta2.T))#5000*10
    p = np.argmax(temp2, axis=1)+1
    return p


pred = predict(theta1, theta2, X)
print('Training Set Accuracy: ', np.sum(pred == Y)/np.size(Y, 0))
_ = input('Press [Enter] to continue.')

# 随机展示图像
num = 10
rindex = np.random.permutation(m)
for i in range(num):
    print('Displaying Example Image')
    displayData(X[rindex[i]:rindex[i]+1, :])

    pred = predict(theta1, theta2, X[rindex[i]:rindex[i]+1, :])
    print('Neural Network Prediction: %d (digit %d)' % (pred, pred % 10))
    _ = input('Press [Enter] to continue.')






