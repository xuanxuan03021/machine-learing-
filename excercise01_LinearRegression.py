import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as op

from mpl_toolkits.mplot3d import Axes3D

#print(np.eye(5))
data=np.loadtxt("ex1data1.txt",delimiter=",")
x=data[:,0]
y=data[:,1]
plt.scatter(x,y)
plt.xlabel('Population of city in 10000s')
plt.ylabel("Profit in $10000")
plt.show()

print('Running Gradient Descent ...')
theta = np.zeros(2)
print(theta)
m = len(y)
print("正常的\n",y)
x=x.reshape(m,1)
y=y.reshape(m,1)
print("reshape后的",y)
x=np.hstack((np.ones((m,1)),x))#在X中加入拼接列,注意hstack和vstack 双括号
#print(x.shape)
theta=theta.reshape(2,1)#否则他不会指定列数为一，就会在之后产生问题
print(theta)

# compute and display initial cost
def computeCost(x,y,theta):
    J = 0
    y=y.reshape(97,1)
    h_of_x=x.dot(theta)-y#矩阵乘法
    #方法一
    #J=(h_of_x.T).dot(h_of_x)/(2*m)
    #方法二
    J=sum(h_of_x*h_of_x)/(2*m)
    return J

def gradientDescent(X, y, theta, alpha, num_iters):
        # Initialize some useful values
        J_history = []
        m = y.size  # number of training examples

        for i in range(num_iters):
            h_of_x = (X.dot(theta) - y).T#  1*97 x=97*2
            h_of_x2=alpha*(h_of_x.dot(X))/m#1*2
            theta=theta-h_of_x2.T
            J_history.append(computeCost(X, y, theta))

        return theta, J_history

J = computeCost(x, y, theta)
print('cost: %0.4f ' % J)

# Some gradient descent settings
iterations = 1500
alpha = 0.01
print(theta)
# run gradient descent
theta, J_history = gradientDescent(x, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent: ')
print('%s %s \n' % (theta[0], theta[1]))

# Plot the linear fit
t1=x[:,1].reshape(m,1)
print(y.shape)
plt.scatter(t1,y,label='training data')#可以将两种线化在同一个图上
plt.plot(t1, x.dot(theta), '-', label='Linear regression')
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
plt.show()


# Predict values for population sizes of 35,000 and 70,000
predict1 = np.array([1, 3.5]).dot(theta)
predict2 = np.array([1, 7]).dot(theta)
print ('For population = 35,000, we predict a profit of ',predict1*10000)
print ('For population = 70,000, we predict a profit of ',predict2*10000)
print('Visualizing J(theta_0, theta_1) ...')


# ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...')
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

J_vals = np.zeros((np.size(theta0_vals, 0), np.size(theta1_vals, 0)))

for i in range(np.size(theta0_vals, 0)):
    for j in range(np.size(theta1_vals, 0)):
        t = np.array([theta0_vals[i], theta1_vals[j]]).reshape(2,1)
        J_vals[i, j] = computeCost(x, y, t)

# 绘制三维图像
theta0_vals, theta1_vals = np.meshgrid(theta0_vals, theta1_vals)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(theta0_vals, theta1_vals, J_vals.T)
ax.set_xlabel(r'$\theta$0')
ax.set_ylabel(r'$\theta$1')

# 绘制等高线图
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# ax2.contour(theta0_vals, theta1_vals, J_vals.T, np.logspace(-2, 3, 20))
# ax2.plot(theta[0], theta[1], 'rx', ms=10, lw=2)
# ax2.set_xlabel(r'$\theta$0')
# ax2.set_ylabel(r'$\theta$1')
plt.show()
