import numpy as np
import matplotlib.pyplot as plt
# ================ Part 1: Feature Normalization ================

def featureNormalize(x):
        mean=np.mean(x,axis=0)
        standard=np.std(x,axis=0)
        x=(x-mean)/standard
        return x,mean,standard

print('Loading data ...')

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
print(X)
X=X.reshape(47,2)
print(X)
y = data[:, 2]
m = y.T.size


# Print out some data points
print('First 10 examples from the dataset:')
#print( np.column_stack( (X[:10], y[:10]) ))

# Scale features and set them to zero mean
print('Normalizing Features ...')

X, mu, sigma = featureNormalize(X)
print('[mu] [sigma]')
print(mu, sigma)

# Add intercept term to X
X = np.concatenate((np.ones((m, 1)), X), axis=1)
#print(X)

# ================ Part 2: Gradient Descent ================
#
# ====================== YOUR CODE HERE ======================
# Instructions: We have provided you with the following starter
#               code that runs gradient descent with a particular
#               learning rate (alpha).
#
#               Your task is to first make sure that your functions -
#               computeCost and gradientDescent already work with
#               this starter code and support multiple variables.
#
#               After that, try running gradient descent with
#               different values of alpha and see which one gives
#               you the best result.
#
#               Finally, you should complete the code at the end
#               to predict the price of a 1650 sq-ft, 3 br house.
#
# Hint: By using the 'hold on' command, you can plot multiple
#       graphs on the same figure.
#
# Hint: At prediction, make sure you do the same feature normalization.
def computeCost(x,y,theta):
    J= 0
    y=y.reshape(47,1)

    h_of_x=x.dot(theta)-y#矩阵乘法 #47*1
    #print(h_of_x.shape)
    #方法一
    J= (h_of_x.T).dot(h_of_x)/(2*m)
    #方法二
    #J=sum(h_of_x*h_of_x)/(2*m)
   # print("the value of J is {}".format(J))
 #   print(J[0][0])
    return J[0][0]



def gradientDescentMulti(X, y, theta, alpha, num_iters):
        # Initialize some useful values
        J_history = []
        m = y.size  # number of training examples

        for i in range(num_iters):
            h_of_x = (X.dot(theta) - y).T#  1*47 x=47*3
            h_of_x2=alpha*(h_of_x.dot(X))/m#1*3
            theta=theta-h_of_x2.T
            J_history.append(computeCost(X, y, theta))

        return theta, J_history

print('Running gradient descent ...')

# Choose some alpha value
alpha = 0.01
num_iters = 400
y=y.reshape(47,1)

# Init Theta and Run Gradient Descent
theta = np.zeros(3).reshape(3,1)
theta, J_history = gradientDescentMulti(X, y, theta, alpha, num_iters)




# Plot the convergence graph
plt.plot(np.arange(np.size(J_history, 0)),J_history)
plt.xlabel('Number of iterations')
plt.ylabel('Cost J')
plt.show()


# Display gradient descent's result
print('Theta computed from gradient descent: ')
print(theta)



# Estimate the price of a 1650 sq-ft, 3 br house


print(mu[0],mu[1])
price = np.array([1,(1650-mu[0])/sigma[0],(3-mu[1])/sigma[1]]).dot(theta)

print('Predicted price of a 1650 sq-ft, 3 br house')
print('(using gradient descent): ')
print(price)


# ================ Part 3: Normal Equations ================

# ====================== YOUR CODE HERE ======================
# Instructions: The following code computes the closed form
#               solution for linear regression using the normal
#               equations. You should complete the code in
#               normalEqn.m
#
#               After doing so, you should complete this code
#               to predict the price of a 1650 sq-ft, 3 br house.
#
def normalEqn(X, y):
    theta = np.linalg.pinv(X.T.dot(X)).dot(X.T).dot(y)
    return theta

print('Solving with normal equations...')

# Load Data
data = np.loadtxt('ex1data2.txt', delimiter=',')
X = data[:, :2]
y = data[:, 2]
m = y.T.size

# Add intercept term to X
X = np.concatenate((np.ones((m,1)), X), axis=1)

# Calculate the parameters from the normal equation
theta = normalEqn(X, y)

# Display normal equation's result
print('Theta computed from the normal equations:')
print(' %s \n' % theta)

# Estimate the price of a 1650 sq-ft, 3 br house
price = np.array([1, 1650, 3]).dot(theta)
print(price)

# ============================================================
