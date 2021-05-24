import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.optimize import minimize
import math
import scipy.linalg as slin

import scipy.optimize as op

np.set_printoptions(threshold = 1e6)#设置打印数量的阈值

## Setup the parameters you will use for this exercise
input_layer_size  = 4 # 4 features
hidden_layer_size = 10   # set your hidden layyer unit
num_labels = 3          # 3 labels, from 1 to 3

iris=np.loadtxt("iris3.txt",delimiter = ",",dtype=str)
X=iris[:,0:4].astype(float)# 0:3取不到3
Y=iris[:,4]
m=np.size(X,axis=0)
#将种类编码成1，2，3
cat1=np.where(Y=="Iris-setosa")
cat2=np.where(Y=="Iris-versicolor")
cat3=np.where(Y=="Iris-virginica")
Y[cat1]=1
Y[cat2]=2
Y[cat3]=3
print(Y)
y=Y.astype(int)
print(y)
Y=Y.reshape(150,1).astype(int)


print('Initializing Neural Network Parameters ...')
def randInitializeWeights(input_layer_size, hidden_layer_size):
    epsilon_init = 0.12
    w=np.random.rand(hidden_layer_size,input_layer_size+1)*2*epsilon_init-epsilon_init
    return w

#初始化参数
initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.hstack((initial_Theta1.T.ravel(), initial_Theta2.T.ravel()))
print(initial_nn_params.shape)
_ = input('Press [Enter] to continue.')

#sigmoid 函数
def sigmoid(z):
    g = 1/(1+np.exp(-1*z))
    return g
#sigmiod的导数

def sigmoidGradient(data):
    sigmoid=1/(1+np.exp(-1*data))
    derivative=sigmoid*(1-sigmoid)
    return derivative

## 损失函数
def nnCost(params, input_layer_size, hidden_layer_size, num_labels, x, y, lamb):
    theta1 = params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)
    m = np.size(x, 0)

    # 前向传播

    a1 = np.concatenate((np.ones((m, 1)), x), axis=1)
    z2 = a1.dot(theta1.T);
    l2 = np.size(z2, 0)
    a2 = np.concatenate((np.ones((l2, 1)), sigmoid(z2)), axis=1)
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
    yt = np.zeros((m, num_labels))
    yt[np.arange(m), y-1] = 1
    #损失函数
    j = np.sum(-yt * np.log(a3) - (1 - yt) * np.log(1 - a3))
    # 向后传播
    j = j / m
    reg_cost = np.sum(np.power(theta1[:, 1:], 2)) + np.sum(np.power(theta2[:, 1:], 2))
    j = j + 1 / (2 * m) * lamb * reg_cost
    return j

# 梯度函数
def nnGrad(params, input_layer_size, hidden_layer_size, num_labels, x, y, lamb):
    theta1 = params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = params[(hidden_layer_size * (input_layer_size + 1)):].reshape(num_labels, hidden_layer_size + 1)
    m = np.size(x, 0)
    # 前向传播 --- 下标：0代表1， 9代表10
    a1 = np.concatenate((np.ones((m, 1)), x), axis=1)
    z2 = a1.dot(theta1.T);
    l2 = np.size(z2, 0)
    a2 = np.concatenate((np.ones((l2, 1)), sigmoid(z2)), axis=1)
    z3 = a2.dot(theta2.T)
    a3 = sigmoid(z3)
    yt = np.zeros((m, num_labels))
    yt[np.arange(m), y - 1] = 1
    # 向后传播
    delta3 = a3 - yt
    delta2 = delta3.dot(theta2) * sigmoidGradient(np.concatenate((np.ones((l2, 1)), z2), axis=1))
    theta2_grad = delta3.T.dot(a2)
    theta1_grad = delta2[:, 1:].T.dot(a1)

    theta2_grad = theta2_grad / m
    theta2_grad[:, 1:] = theta2_grad[:, 1:] + lamb / m * theta2[:, 1:]
    theta1_grad = theta1_grad / m
    theta1_grad[:, 1:] = theta1_grad[:, 1:] + lamb / m * theta1[:, 1:]

    grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()))
    return grad

Lambda=1
j=nnCost(initial_nn_params, input_layer_size, hidden_layer_size,
    num_labels, X, y, Lambda)


_ = input('Press [Enter] to continue.')
print('Training Neural Network...')
#正则化系数
lamb = 1
param = op.fmin_cg(nnCost, initial_nn_params, fprime=nnGrad, \
                    args=(input_layer_size, hidden_layer_size, num_labels, X, y, lamb), maxiter=50)
#拟合后的系数
theta1 = param[0: hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size, input_layer_size+1)
theta2 = param[hidden_layer_size*(input_layer_size+1):].reshape(num_labels, hidden_layer_size+1)
_ = input('Press [Enter] to continue.')

#回归结果精度

def predict(Theta1, Theta2, X):

    X=np.hstack((np.ones((m,1)),X))
    temp1=sigmoid(X.dot(Theta1.T))
    temp = np.concatenate((np.ones((m, 1)), temp1), axis=1)#5000*26
    temp2 = sigmoid(temp.dot(Theta2.T))#5000*10
    p=np.argmax(temp2,axis=1)+1
    return p


pred = predict(theta1, theta2, X)

print(pred)
accuracy =np.sum(pred == y)/np.size(y,axis=0)
print('Training Set Accuracy: %f\n'% accuracy)
