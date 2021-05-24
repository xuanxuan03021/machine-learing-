import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as op

def sigmoid(z):
    g =1/(1+np.exp(-z))
    return g


def costFunctionReg(theta,x,y,lam):
    m=len(y)
    h_of_x=sigmoid(x.dot(theta))#100*1
    J=(((-1*y).T).dot(np.log(h_of_x))-((1-y).T).dot(np.log(1-h_of_x)))/m+lam/(2*m)*(initial_theta[1:,].dot(initial_theta[1:,]))#一维可以不用转置 （18，）、（18，0）直接点成也可以
    return J

def Plot(x,y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    plt.scatter(x[pos,0], x[pos, 1], marker='+', s=50, color='b',label="admitted")
    plt.scatter(x[neg, 0], x[neg, 1], marker='o', s=50, color='r',label="no_admitted")
    plt.legend(loc='upper right', fontsize=8)
    plt.xlabel("Microchip Test 1")
    plt.xlabel("Microchip Test 2")
    plt.show()
def mapFeature(x1,x2):
    degree=6
    m=len(x1)
    out=np.ones((m,28))
    count=1
    for i in range(1,degree+1):
        for j in range(i+1):
            out[:, count] = np.power(x1, i - j) * np.power(x2, j)
            count += 1
    return out

# def gradientFunction(theta,x,y,lam):
#     m=len(y)
#     cost=np.zeros((1,1))
#     h_of_x = sigmoid(x.dot(theta))
#     print(h_of_x.shape)#118*1
#     print(x[:,0].shape)#118*28
#     print((x[:,1:].T).dot(h_of_x-y)/m+lam*theta[1:]/m)
#     cost[0]=(x[:,0].T).dot(h_of_x-y)/m
#     cost[1:]=(x[:,1:].T).dot(h_of_x-y)/m+lam*theta[1:]/m
#     return cost

def gradFuncReg(theta, x, y, lam):
    m = np.size(y, 0)
    h = sigmoid(x.dot(theta))
    grad = np.zeros(np.size(theta, 0))
    grad[0] = 1/m*(x[:, 0].dot(h-y))
    grad[1:] = 1/m*(x[:, 1:].T.dot(h-y))+lam*theta[1:]/m
    return grad

data = np.loadtxt('exc2data2.txt', delimiter=',')
x=data[:,0:2]
y=data[:,2]
Plot(x,y)
x=mapFeature(x[:,0],x[:,1])
m=np.size(x,1)
print(m)
#initial_theta = np.zeros((m ,1))#这样会出错

initial_theta = np.zeros((m ,))
print(initial_theta)

init_theta2 = np.zeros((m,1))
print(init_theta2)

# Set regularization parameter lambda to 1
Lambda = 1

# Compute and display initial cost and gradient for regularized logistic
# regression
print(initial_theta.T.dot(initial_theta))
cost = costFunctionReg(initial_theta, x, y, Lambda)
print('Cost at initial theta (zeros): %f' % cost)
gran=gradFuncReg(initial_theta,x,y,Lambda)
result = op.minimize(costFunctionReg, x0=initial_theta, method='BFGS', jac=gradFuncReg, args=(x, y,Lambda))
theta = result.x


def plotDecisionBoundary(theta, x, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1 = plt.scatter(x[pos, 1], x[pos, 2], marker='+', s=60, color='r')
    p2 = plt.scatter(x[neg, 1], x[neg, 2], marker='o', s=60, color='y')
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)
    z = np.zeros((np.size(u, 0), np.size(v, 0)))
    for i in range(np.size(u, 0)):
        for j in range(np.size(v, 0)):
            z[i, j] = mapFeature(np.array([u[i]]), np.array([v[j]])).dot(theta)
    z = z.T
    [um, vm] = np.meshgrid(u, v)
    #绘制等高线
    plt.contour(um, vm, z, levels=[0], lw=2)
    plt.legend((p1, p2), ('Admitted', 'Not admitted'), loc='upper right', fontsize=8)
    plt.xlabel('Microchip Test 1')
    plt.ylabel('Microchip Test 2')
    plt.title('lambda = 1')
    plt.show()

plotDecisionBoundary(theta, x, y)

# 预测给定值
def predict(theta, x):
    m = np.size(x, 0)
    p = np.zeros((m,))
    pos = np.where(x.dot(theta) >= 0)
    neg = np.where(x.dot(theta) < 0)
    p[pos] = 1
    p[neg] = 0
    return p

p = predict(theta, x)
print('Train Accuracy: ', np.sum(p == y)/np.size(y, 0))