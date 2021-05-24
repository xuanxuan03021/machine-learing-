import matplotlib.pyplot as plt
import numpy as np
import scipy.io
from scipy.optimize import minimize
import math
import scipy.linalg as slin

import scipy.optimize as op

np.set_printoptions(threshold = 1e6)#设置打印数量的阈值

## Setup the parameters you will use for this exercise
input_layer_size  = 400  # 20x20 Input Images of Digits
hidden_layer_size = 25   # 25 hidden units
num_labels = 10          # 10 labels, from 1 to 10   
                          # (note that we have mapped "0" to label 10)
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

## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset. 
#  You will be working with a dataset that contains handwritten digits.
#

# Load Training Data
print('Loading and Visualizing Data ...')

data = scipy.io.loadmat('ex4data1.mat')
X = data['X']
y = data['y'][:, 0]
m=np.size(X,axis=0)

# Randomly select 100 data points to display
rand_indices = np.random.permutation(range(m))
sel = X[rand_indices[0:100], :]

displayData(sel)


## ================ Part 2: Loading Parameters ================
# In this part of the exercise, we load some pre-initialized 
# neural network parameters.

print('Loading Saved Neural Network Parameters ...')

# Load the weights into variables Theta1 and Theta2
data = scipy.io.loadmat('ex4weights.mat')
Theta1 = data['Theta1']#25*401
Theta2 = data['Theta2']#10*26
Theta=np.concatenate((Theta1.flatten(), Theta2.flatten()))

## ================ Part 3: Compute Cost (Feedforward) ================
#  To the neural network, you should first start by implementing the
#  feedforward part of the neural network that returns the cost only. You
#  should complete the code in nnCostFunction.m to return cost. After
#  implementing the feedforward to compute the cost, you can verify that
#  your implementation is correct by verifying that you get the same cost
#  as us for the fixed debugging parameters.
#
#  We suggest implementing the feedforward cost *without* regularization
#  first so that it will be easier for you to debug. Later, in part 4, you
#  will get to implement the regularized cost.
#


print('Feedforward Using Neural Network ...')

# Weight regularization parameter (we set this to 0 here).
Lambda = 0
def sigmoid(z):
    g = 1/(1+np.exp(-1*z))
    return g

def sigmoidGradient(data):
    sigmoid=1/(1+np.exp(-1*data))
    derivative=sigmoid*(1-sigmoid)
    return derivative

def nnCostFunction(Theta, input_layer_size, hidden_layer_size,
    num_labels, X, Y, Lambda):
    Theta1 = Theta[0:hidden_layer_size*(input_layer_size+1)].reshape((hidden_layer_size, input_layer_size+1))
    Theta2 = Theta[(hidden_layer_size*(input_layer_size+1)):].reshape((num_labels, hidden_layer_size+1))
    m=np.size(X,axis=0)
#改变y的类别
    y = np.zeros((m, num_labels))
    y[np.arange(m), Y-1] = 1

    X=np.hstack((np.ones((m,1)),X))#5000*401
    temp1=sigmoid(X.dot(Theta1.T))#5000*25
    temp = np.concatenate((np.ones((m, 1)), temp1), axis=1)#5000*26
    temp2 = sigmoid(temp.dot(Theta2.T))#5000*10
    J_theta_temp=((-1*y)*(np.log(temp2))-(1-y)*(np.log(1-temp2)))
    J_theta=1/m*np.sum(J_theta_temp)+(Lambda/(2*m))*(np.sum(np.power(Theta1[:, 1:], 2)) + np.sum(np.power(Theta2[:, 1:], 2)))


    # 反向传播 --- 下标：0代表1， 9代表10
    a1 = X#5000*401
    z2 = a1.dot(Theta1.T); l2 = np.size(z2, 0)#5000*25
    a2 = np.concatenate((np.ones((l2, 1)), sigmoid(z2)), axis=1)#5000*26
    z3 = a2.dot(Theta2.T)#5000*10
    a3 = sigmoid(z3)

    delta3=a3-y# 5000*10
    delta2=delta3.dot(Theta2)*sigmoidGradient(np.concatenate((np.ones((l2, 1)), z2), axis=1))#26*1
    theta2_grad = delta3.T.dot(a2)#10*26
    theta1_grad = delta2[:, 1:].T.dot(a1)

    theta2_grad = theta2_grad / m #权重的导数

    theta2_grad[:, 1:] = theta2_grad[:, 1:]+Lambda/m*Theta2[:, 1:]
    theta1_grad = theta1_grad / m
    theta1_grad[:, 1:] = theta1_grad[:, 1:] + Lambda/ m * Theta1[:, 1:]
    grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()))


    return J_theta,grad



# Y=np.zeros((5000,10))
# #类别转变，将1，2。。。10 转换成题目要求
# for i in range(5000):
#     cat=y[i,]
#     Y[i,cat-1]=1


J,_= nnCostFunction(Theta, input_layer_size, hidden_layer_size,
    num_labels, X, y, Lambda)

print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.287629)\n' % J)

_=input("Program paused. Press Enter to continue...")

## =============== Part 4: Implement Regularization ===============
#  Once your cost function implementation is correct, you should now
#  continue to implement the regularization with the cost.
#

print('Checking Cost Function (w/ Regularization) ...')

# Weight regularization parameter (we set this to 1 here).
Lambda = 1

J,_= nnCostFunction(Theta, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

print('Cost at parameters (loaded from ex4weights): %f \n(this value should be about 0.383770)' % J)

_ = input('Press [Enter] to continue.')

## ================ Part 5: Sigmoid Gradient  ================
#  Before you start implementing the neural network, you will first
#  implement the gradient for the sigmoid function. You should complete the
#  code in the sigmoidGradient.m file.
#



print('Evaluating sigmoid gradient...')

g = sigmoidGradient(np.array([1, -0.5, 0, 0.5, 1]))
print ('Sigmoid gradient evaluated at [1 -0.5 0 0.5 1]: ')
print(g)


_ = input('Press [Enter] to continue.')

## ================ Part 6: Initializing Pameters ================
#  In this part of the exercise, you will be starting to implment a two
#  layer neural network that classifies digits. You will start by
#  implementing a function to initialize the weights of the neural network
#  (randInitializeWeights.m)

print('Initializing Neural Network Parameters ...')
def randInitializeWeights(input_layer_size, hidden_layer_size):
    epsilon_init = 0.12
    w=np.random.rand(hidden_layer_size,input_layer_size+1)*2*epsilon_init-epsilon_init
    return w


initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size)
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels)

# Unroll parameters
initial_nn_params = np.hstack((initial_Theta1.T.ravel(), initial_Theta2.T.ravel()))
_ = input('Press [Enter] to continue.')

## =============== Part 7: Implement Backpropagation ===============
#  Once your cost matches up with ours, you should proceed to implement the
#  backpropagation algorithm for the neural network. You should add to the
#  code you've written in nnCostFunction.m to return the partial
#  derivatives of the parameters.
#


print('Checking Backpropagation... ')
# 调试时的参数初始化
def debugInitWeights(fout, fin):
    w = np.sin(np.arange(fout*(fin+1))+1).reshape(fout, fin+1)/10
    return w

# 数值法计算梯度
def computeNumericalGradient(J, theta, args):
    numgrad = np.zeros(np.size(theta))
    perturb = np.zeros(np.size(theta))
    epsilon = 1e-4
    for i in range(np.size(theta)):
        perturb[i] = epsilon
        loss1, _ = J(theta-perturb, *args)#将args大散成多个参数
        loss2, _ = J(theta+perturb, *args)
        numgrad[i] = (loss2-loss1)/(2*epsilon)
        perturb[i] = 0
    return numgrad
# 检查神经网络的梯度
def checkNNGradients(lamb):
    input_layer_size = 3
    hidden_layer_size = 5
    num_labels = 3
    m = 5

    theta1 = debugInitWeights(hidden_layer_size, input_layer_size)
    theta2 = debugInitWeights(num_labels, hidden_layer_size)

    x = debugInitWeights(m, input_layer_size-1)
    y = 1+(np.arange(m)+1) % num_labels

    nn_params = np.concatenate((theta1.flatten(), theta2.flatten()))

    cost, grad = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, x, y, lamb)
    numgrad = computeNumericalGradient(nnCostFunction, nn_params,\
                                       (input_layer_size, hidden_layer_size, num_labels, x, y, lamb))
    print('The above two columns you get should be very similar.\n \
    (Left-Your Numerical Gradient, Right-Analytical Gradient)')
    diff = slin.norm(numgrad-grad)/slin.norm(numgrad+grad)
    print('If your backpropagation implementation is correct, then \n\
         the relative difference will be small (less than 1e-9). \n\
         \nRelative Difference: ', diff)

#  Check gradients by running checkNNGradients
checkNNGradients(0)

_ = input('Press [Enter] to continue.')

## =============== Part 8: Implement Regularization ===============
#  Once your backpropagation implementation is correct, you should now
#  continue to implement the regularization with the cost and gradient.
#

print('Checking Backpropagation (w/ Regularization) ... ')

#  Check gradients by running checkNNGradients
Lambda = 3.0
checkNNGradients(Lambda)

# Also output the costFunction debugging values
debug_J, _ = nnCostFunction(Theta, input_layer_size, hidden_layer_size, num_labels, X, y, Lambda)

print('Cost at (fixed) debugging parameters (w/ lambda = 10): %f (this value should be about 0.576051)\n\n' % debug_J)

_ = input('Press [Enter] to continue.')


## =================== Part 8: Training NN ===================
#  You have now implemented all the code necessary to train a neural
#  network. To train your neural network, we will now use "fmincg", which
#  is a function which works similarly to "fminunc". Recall that these
#  advanced optimizers are able to train our cost functions efficiently as
#  long as we provide them with the gradient computations.
#
print ('Training Neural Network... ')

#  After you have completed the assignment, change the MaxIter to a larger
#  value to see how more training helps.
# options = optimset('MaxIter', 50)

## 损失函数
def nnCost(params, input_layer_size, hidden_layer_size, num_labels, x, y, lamb):
    theta1 = params[0:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, input_layer_size + 1)
    theta2 = params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, hidden_layer_size + 1)
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

print('Training Neural Network...')
lamb = 1
param = op.fmin_cg(nnCost, initial_nn_params, fprime=nnGrad,args=(input_layer_size, hidden_layer_size, num_labels, X, y, lamb), maxiter=50)

theta1 = param[0: hidden_layer_size*(input_layer_size+1)].reshape(hidden_layer_size, input_layer_size+1)
theta2 = param[hidden_layer_size*(input_layer_size+1):].reshape(num_labels, hidden_layer_size+1)
_ = input('Press [Enter] to continue.')

#
# ## ================= Part 9: Visualize Weights =================
# #  You can now "visualize" what the neural network is learning by
# #  displaying the hidden units to see what features they are capturing in
# #  the data.
#
# print 'Visualizing Neural Network... '
#
# displayData(Theta1[:, 1:])
#
# raw_input("Program paused. Press Enter to continue...")
#
## ================= Part 10: Implement Predict =================
#  After training the neural network, we would like to use it to predict
#  the labels. You will now implement the "predict" function to use the
#  neural network to predict the labels of the training set. This lets
#  you compute the training set accuracy.]


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


