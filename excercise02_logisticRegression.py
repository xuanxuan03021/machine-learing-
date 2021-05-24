# Logistic Regression
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op



## Machine Learning Online Class - Exercise 2: Logistic Regression
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the second part
#  of the exercise which covers regularization with logistic regression.
#
#  You will need to complete the following functions in this exericse:
#
#     sigmoid.py
#     costFunction.py
#     gradientFunction.py
#     predict.py
#     costFunctionReg.py
#     gradientFunctionReg.py
#     n.b. This files differ in number from the Octave version of ex2.
#          This is due to the scipy optimization taking only scalar
#          functions where fmiunc in Octave takes functions returning
#          multiple values.
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
# Load Data
#  The first two columns contains the exam scores and the third column
#  contains the label.

data = np.loadtxt('ex2data1.txt', delimiter=',')
X = data[:, 0:2]
y = data[:, 2]
X=X.reshape(100,2)
y=y.reshape(100,1)
# ==================== Part 1: Plotting ====================

print ('Plotting data with + indicating (y = 1) examples and o indicating (y = 0) examples.')

def plotData(x, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1 = plt.scatter(x[pos, 0], x[pos, 1], marker='+', s=30, color='b')
    p2 = plt.scatter(x[neg, 0], x[neg, 1], marker='o', s=30, color='y')
    plt.legend((p1, p2), ('Admitted', 'Not admitted'), loc='upper right', fontsize=8)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()


def sigmoid(z):
    g =1/(1+np.exp(-z))
    return g


def costFunction(theta,x,y):
    m=len(y)
    print(m)
    h_of_x=sigmoid(x.dot(theta))#100*1
    J=(((-1*y).T).dot(np.log(h_of_x))-((1-y).T).dot(np.log(1-h_of_x)))/m
    return J

plotData(X, y)

# ============ Part 2: Compute Cost and Gradient ============

def gradientFunction(theta,x,y):
    m=len(y)
    h_of_x = sigmoid(x.dot(theta))
    cost=(x.T).dot(h_of_x-y)/m
    return cost

#  Setup the data matrix appropriately, and add ones for the intercept term
m, n = X.shape

# Add intercept term to x and X_test
X = np.concatenate((np.ones((m, 1)), X), axis=1)

# Initialize fitting parameters
initial_theta = np.zeros((n + 1,1))

# Compute and display initial cost and gradient
cost = costFunction(initial_theta, X, y)
print('Cost at initial theta (zeros): %f' % cost)

grad = gradientFunction(initial_theta, X, y)
print('Gradient at initial theta (zeros): ' + str(grad))

__=input("Program paused. Press Enter to continue...")

# ============= Part 3: Optimizing using scipy  =============
#官方文档https://docs.scipy.org/doc/scipy/reference/optimize.html
res = op.minimize(costFunction, initial_theta, method='TNC',jac=False, args=(X, y), options={'gtol': 1e-3, 'disp': True, 'maxiter': 1000})
result = op.minimize(costFunction, x0=initial_theta, method='BFGS', jac=gradientFunction, args=(X, y))
#最小化的参数必须写在第一个位置 costfunction 中的参数里theta必须在第一位
#costFunction 需要最小化的参数
#method 追最小化的方法，仿佛都差不多
#jac 怎么求的梯度 自己写的梯度方程
#arg costFunction 里的参数
#options 里有选项 GTOL 梯度的误差 disp 展示信息 maxiter：最大迭代数

theta = result.x
theta = res.x
cost = res.fun

# Print theta to screen
print ('Cost at theta found by scipy: %f' % cost)
print( 'theta:', ["%0.4f" % i for i in theta])

#Plotting the decision boundary: two points, draw a line between
#Decision boundary occurs when h = 0, or when
#theta0 + theta1*x1 + theta2*x2 = 0
#y=mx+b is replaced by x2 = (-1/thetheta2)(theta0 + theta1*x1)
def plotDecisionBoundary(theta, x, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p1 = plt.scatter(x[pos, 1], x[pos, 2], marker='+', s=60, color='r')
    p2 = plt.scatter(x[neg, 1], x[neg, 2], marker='o', s=60, color='y')
    plot_x = np.array([np.min(x[:, 1])-2, np.max(x[:, 1]+2)])
    plot_y = -1/theta[2]*(theta[1]*plot_x+theta[0])
    plt.plot(plot_x, plot_y)
    plt.legend((p1, p2), ('Admitted', 'Not admitted'), loc='upper right', fontsize=8)
    plt.xlabel('Exam 1 score')
    plt.ylabel('Exam 2 score')
    plt.show()

# Plot Boundary
plotDecisionBoundary(theta, X, y)

# Labels and Legend
plt.legend(['Admitted', 'Not admitted'], loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.show()

_=input("Program paused. Press Enter to continue...")

# ============== Part 4: Predict and Accuracies ==============
prob = sigmoid(np.array([1, 45, 85]).dot(theta))
print('For a student with scores 45 and 85, we predict an admission probability of: ', prob)

# 预测给定值
def predict(theta, x):
    m = np.size(X, 0)
    p = np.zeros((m,))
    pos = np.where(x.dot(theta) >= 0)
    neg = np.where(x.dot(theta) < 0)
    p[pos] = 1
    p[neg] = 0
    return p
p = predict(theta, X)
print('Train Accuracy: ', np.sum(p ==y)/np.size(y, 0))
_ = input('Press [Enter] to continue.')
p=predict(theta, X)