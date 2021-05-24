import scipy.io
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as op


## Machine Learning Online Class
#  Exercise 5 | Regularized Linear Regression and Bias-Variance
## =========== Part 1: Loading and Visualizing Data =============
#  We start the exercise by first loading and visualizing the dataset.
#  The following code will load the dataset into your environment and plot
#  the data.
#

# Load Training Data
print('Loading and Visualizing Data ...')

# Load from ex5data1:
# You will have X, y, Xval, yval, Xtest, ytest in your environment
data = scipy.io.loadmat('ex5data1.mat')

# m = Number of examples
X = data['X'][:, 0]
y = data['y'][:, 0]
Xval = data['Xval'][:, 0]
yval = data['yval'][:, 0]
Xtest = data['Xtest'][:, 0]

m = X.size
#Plot training data
plt.scatter(X, y, marker='x', s=60, edgecolor='r', lw=1.5)
plt.ylabel('Water flowing out of the dam (y)')            # Set the y-axis label
plt.xlabel('Change in water level (x)')     # Set the x-axis label
plt.show()
_=input("Program paused. Press Enter to continue...")

## =========== Part 2: Regularized Linear Regression Cost =============
#  You should now implement the cost function for regularized linear
#  regression.
#

def linearRegCostFunction(theta,X,y,lamb):
    h_of_x=X.dot(theta)
    m = np.size(y, 0)
    grad = np.zeros(np.shape(theta))
    J=1/(2*m)*(h_of_x-y).dot(h_of_x-y)+lamb/(2*m)*theta[1:].dot(theta[1:].T)
    grad[0]=1/m*(h_of_x-y).dot(X[:,0])
    grad[1:]=1/m*(h_of_x-y).dot(X[:,1:])+lamb/m*theta[1:]
    return J

def gradFuncReg(theta,X,y,lamb):
    h_of_x=X.dot(theta)
    m = np.size(y, 0)
    grad = np.zeros(np.shape(theta))
    grad[0]=1/m*(h_of_x-y).dot(X[:,0])
    grad[1:]=1/m*(h_of_x-y).dot(X[:,1:])+lamb/m*theta[1:]
    return grad


print("*"*200)
print(np.column_stack((np.ones(m), X)))
theta = np.array([1, 1])
print(theta.shape)
J = linearRegCostFunction(theta,np.column_stack((np.ones(m), X)), y, 1)

print('Cost at theta = [1  1]: %f \n(this value should be about 303.993192)\n' % J)


_=input("Program paused. Press Enter to continue...")

## =========== Part 3: Regularized Linear Regression Gradient =============
#  You should now implement the gradient for regularized linear
#  regression.
#
grad= gradFuncReg(theta,np.column_stack((np.ones(m), X)), y, 1)
print('Gradient at theta = [1 ; 1]:  [%f; %f] \
\n(this value should be about [-15.303016; 598.250744])' % (grad[0], grad[1]))




## =========== Part 4: Train Linear Regression =============
#  Once you have implemented the cost and gradient correctly, the
#  trainLinearReg function will use your cost function to train
#  regularized linear regression.
#
#  Write Up Note: The data is non-linear, so this will not give a great
#                 fit.
#
def trainLinearReg(X,y,lamb):

   # print(X)
    init_theta = np.zeros((np.size(X, 1),))
   # print(init_theta)
    result = op.minimize(linearRegCostFunction, x0=init_theta, method='BFGS', jac=gradFuncReg, args=(X, y, lamb))
    return result.x
#  Train linear regression with Lambda = 0

Lambda = 0
X=X.reshape(12,1)

x_=np.hstack((np.ones((m,1)),X))
result = trainLinearReg(x_, y, Lambda)
print("result:",result)

#  Plot fit over the data
plt.scatter(X, y, marker='x', s=20, edgecolor='r', lw=1.5)
plt.ylabel('Water flowing out of the dam (y)')            # Set the y-axis label
plt.xlabel('Change in water level (x)')     # Set the x-axis label
plt.plot(X, np.column_stack((np.ones(m), X)).dot(theta), lw=2.0)
plt.show()

_=input("Program paused. Press Enter to continue...")


# =========== Part 5: Learning Curve for Linear Regression =============
#  Next, you should implement the learningCurve function.
#
#  Write Up Note: Since the model is underfitting the data, we expect to
#                 see a graph with "high bias" -- slide 8 in ML-advice.pd
def learningCurve(X,Y,X_val,y_val,lamb):
    error_train=[]
    error_val=[]
    for i in range(np.size(Y)):
        train_x=X[:i+1,:]
        train_y=Y[:i+1]
        print(X.shape)
        print(Y.shape)
        theta = trainLinearReg(train_x,train_y,lamb)
        train_error=linearRegCostFunction(theta,train_x,train_y,lamb)
        val_error=linearRegCostFunction(theta,X_val,y_val,lamb)
        error_train.append(train_error)
        error_val.append(val_error)
    return error_train,error_val


Lambda = 0
error_train, error_val = learningCurve(np.column_stack((np.ones(m), X)), y,
                                       np.column_stack((np.ones(Xval.shape[0]), Xval)), yval, Lambda)
plt.figure()
plt.plot(range(m), error_train, color='b', lw=0.5, label='Train')
plt.plot(range(m), error_val, color='r', lw=0.5, label='Cross Validation')
plt.title('Learning curve for linear regression')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim(0, 13)
plt.ylim(0, 150)
plt.legend(loc='upper right', shadow=True, fontsize='x-large', numpoints=1)
plt.show()

print('Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print ('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

_=input("Program paused. Press Enter to continue...")

## =========== Part 6: Feature Mapping for Polynomial Regression =============
#  One solution to this is to use polynomial regression. You should now
#  complete polyFeatures to map each example into its powers
def polyFeatures(X,p):
    print("**")
    print(X.shape)
    m=np.size(X,0)
    x=np.zeros((m,p))
    for i in range(0,p):
        x[:,i]=np.power(X,i+1)
    return x

def featureNormalize(X_poly):

    mu=np.mean(X_poly,axis=0)
    print(mu)
    sigma=np.std(X_poly,axis=0)
    x_nor=(X_poly-mu)/sigma
    print(sigma)
    return x_nor,mu,sigma

p = 8

# Map X onto Polynomial Features and Normalize
X_poly = polyFeatures(X[:,0], p)
X_poly, mu, sigma =featureNormalize(X_poly)  # Normalize
X_poly = np.column_stack((np.ones(m), X_poly))                   # Add Ones

# Map X_poly_test and normalize (using mu and sigma)
X_poly_test = polyFeatures(Xtest, p)
X_poly_test = X_poly_test - mu #要用同样的均值和标准差
X_poly_test = X_poly_test / sigma
X_poly_test = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_test))        # Add Ones

# Map X_poly_val and normalize (using mu and sigma)
X_poly_val = polyFeatures(Xval, p)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
X_poly_val = np.column_stack((np.ones(X_poly_test.shape[0]), X_poly_val))           # Add Ones

print('Normalized Training Example 1:')
print (X_poly[0, :])

print('\nProgram paused. Press enter to continue.')



# =========== Part 7: Learning Curve for Polynomial Regression =============
#  Now, you will get to experiment with polynomial regression with multiple
#  values of Lambda. The code below runs polynomial regression with
#  Lambda = 0. You should try running the code with different values of
#  Lambda to see how the fit and learning curve change.


Lambda = 0
theta = trainLinearReg(X_poly, y, Lambda, method='BFGS', maxiter=10)


def plotFit(x_min,x_max,mu,sigma,theta,p):
    x=np.arange(x_min-15,x_max+25,0.05)
    x_ploy=polyFeatures(x,p)
    x_ploy=(x_ploy-mu)/sigma
    x_ploy=np.hstack((np.size(x,0),x_ploy))
    return x_ploy,x_ploy.dot(theta)


x_poly,y_poly=plotFit(min(X), max(X), mu, sigma, theta, p)
# Plot training data and fit
plt.figure()
plt.scatter(X, y, marker='x', s=10, edgecolor='r', lw=1.5)
plt.plot(x_poly,y_poly,"--")
plt.xlabel('Change in water level (x)')            # Set the y-axis label
plt.ylabel('Water flowing out of the dam (y)')     # Set the x-axis label
# plt.plot(X, np.column_stack((np.ones(m), X)).dot(theta), marker='_',  lw=2.0)
plt.title('Polynomial Regression Fit (Lambda = %f)' % Lambda)
plt.show()


error_train, error_val = learningCurve(X_poly, y, X_poly_val, yval, Lambda)
plt.plot(range(m), error_train, label='Train')
plt.plot(range(m), error_val, label='Cross Validation')
plt.title('Polynomial Regression Learning Curve (Lambda = %f)' % Lambda)
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.xlim(0, 13)
plt.ylim(0, 150)
plt.legend()

print('Polynomial Regression (Lambda = %f)\n\n' % Lambda)
print ('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(m):
    print ('  \t%d\t\t%f\t%f' % (i, error_train[i], error_val[i]))

_=input("Program paused. Press Enter to continue...")

# ## =========== Part 8: Validation for Selecting Lambda =============
# #  You will now implement validationCurve to test various values of
# #  Lambda on a validation set. You will then use this to select the
# #  "best" Lambda value.
# #
#
# Lambda_vec, error_train, error_val = validationCurve(X_poly, y, X_poly_val, yval)
#
# plt.plot(Lambda_vec, error_train, Lambda_vec, error_val)
# plt.legend('Train', 'Cross Validation')
# plt.xlabel('Lambda')
# plt.ylabel('Error')
#
# print 'Lambda\t\tTrain Error\tValidation Error'
# for i in range(Lambda_vec.size):
#     print ' %f\t%f\t%f' % (Lambda_vec[i], error_train[i], error_val[i])
#
# raw_input("Program paused. Press Enter to continue...")