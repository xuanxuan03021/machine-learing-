import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
import scipy.optimize as op


# ## =============== Part 1: Loading and Visualizing Data ================
# #  We start the exercise by first loading and visualizing the dataset.
# #  The following code will load the dataset into your environment and plot
# #  the data.
# #
#
# print('Loading and Visualizing Data ...')
#
# # Load from ex6data1:
# # You will have X, y in your environment
# data = scipy.io.loadmat('ex6data1.mat')
# X = data['X']
# y = data['y'].flatten()
# print(X)
# print(y)
def plotData(X,y):
    pos=np.where(y==1)
    neg=np.where(y==0)
    plt.scatter(X[pos,0],X[pos,1],marker='+',s=60)
    plt.scatter(X[neg,0],X[neg,1],marker='o',s=60)
    plt.show()
#
# # Plot training data
# plotData(X, y)
#
# _=input("Program paused. Press Enter to continue...")
#
# ## ==================== Part 2: Training Linear SVM ====================
# #  The following code will train a linear SVM on the dataset and plot the
# #  decision boundary learned.
# #
#
# # Load from ex6data1:
# # You will have X, y in your environment
# data = scipy.io.loadmat('ex6data1.mat')
# X = data['X']
# y = data['y'].flatten()
#
# print('Training Linear SVM ...')
#
# # You should try to change the C value below and see how the decision
# # boundary varies (e.g., try C = 1000)
# def visualizeBoundaryLinear(X, y, model):
#     """plots a linear decision boundary
#     learned by the SVM and overlays the data on it
#     """
#
#     w = model.coef_.flatten()
#     b = model.intercept_.flatten()
#     xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
#     yp = -(w[0]*xp + b)/w[1]
#     pos=np.where(y==1)
#     neg=np.where(y==0)
#     plt.scatter(X[pos,0],X[pos,1],marker='+',s=60)
#     plt.scatter(X[neg,0],X[neg,1],marker='o',s=60)
#     plt.plot(xp, yp, '-b')
#     plt.show()
#
# C = 1
# clf = svm.SVC(C=C, kernel='linear', tol=1e-3, max_iter=20)
# model = clf.fit(X, y)
# visualizeBoundaryLinear(X, y, model)
#
# _=input("Program paused. Press Enter to continue...")
#
# ## =============== Part 3: Implementing Gaussian Kernel ===============
# #  You will now implement the Gaussian kernel to use
# #  with the SVM. You should complete the code in gaussianKernel.m
# #
# print('Evaluating the Gaussian Kernel ...')
#
# x1 = np.array([1, 2, 1])
# x2 = np.array([0, 4, -1])
# sigma = 2
# def gaussianKernel(x1,x2,sigma):
#     sim = np.exp(-(x1 - x2).dot(x1 - x2) / (2 * sigma ** 2))
#     return sim
#
# sim = gaussianKernel(x1, x2, sigma)
#
# print('Gaussian Kernel between x1 = [1 2 1], x2 = [0 4 -1], sigma = %0.5f : ' \
#        '\t%f\n(this value should be about 0.324652)\n' % (sigma, sim))
#
# _=input("Program paused. Press Enter to continue...")
#
# ## =============== Part 4: Visualizing Dataset 2 ================
# #  The following code will load the next dataset into your environment and
# #  plot the data.
# #
#
# print('Loading and Visualizing Data ...')
#
# # Load from ex6data2:
# # You will have X, y in your environment
# data = scipy.io.loadmat('ex6data2.mat')
# X = data['X']
# y = data['y'].flatten()
#
# # Plot training data
# plotData(X, y)
#
# _=input("Program paused. Press Enter to continue...")
#
## ========== Part 5: Training SVM with RBF Kernel (Dataset 2) ==========
#  After you have implemented the kernel, we can now use it to train the
#  SVM classifier.
#
print=('Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...')

# Load from ex6data2:
# You will have X, y in your environment
data = scipy.io.loadmat('ex6data2.mat')
X = data['X']
y = data['y'].flatten()

# SVM Parameters
C = 1
sigma = 0.1
gamma = 1.0 / (2.0 * sigma ** 2)

# We set the tolerance and max_passes lower here so that the code will run
# faster. However, in practice, you will want to run the training to
# convergence.

def visualizeBoundary(x, y, model):
    pos=np.where(y==1)
    neg=np.where(y==0)
    plt.scatter(X[pos,0],X[pos,1],marker='+',s=60)
    plt.scatter(X[neg,0],X[neg,1],marker='o',s=60)
    x1plot = np.linspace(np.min(x[:, 0]), np.max(x[:, 0]), 100)
    x2plot = np.linspace(np.min(x[:, 1]), np.max(x[:, 1]), 100)
    x1, x2 = np.meshgrid(x1plot, x2plot)
    vals = np.zeros(np.shape(x1))
    for i in range(np.size(x1, 1)):
        this_x = np.vstack((x1[:, i], x2[:, i])).T
        vals[:, i] = model.predict(this_x)
    plt.contour(x1, x2, vals,levels=[0.5],colors='b')
    plt.show()
clf = svm.SVC(C=C, kernel='rbf', tol=1e-3, max_iter=200, gamma=gamma)
model = clf.fit(X, y)
visualizeBoundary(X, y, model)

_=input("Program paused. Press Enter to continue...")

## =============== Part 6: Visualizing Dataset 3 ================
#  The following code will load the next dataset into your environment and
#  plot the data.
#

def print(a):
    """
    :param a: humengxuan
    :return:
    """
    return a

# Load from ex6data3:
# You will have X, y in your environment
# data = scipy.io.loadmat('ex6data3.mat')
# X = data['X']
# y = data['y'].flatten()
#
# # Plot training data
# plotData(X, y)
#
# _=input("Program paused. Press Enter to continue...")

# ## ========== Part 7: Training SVM with RBF Kernel (Dataset 3) ==========

#  This is a different dataset that you can use to experiment with. Try
#  different values of C and sigma here.
#

# Load from ex6data3:
# You will have X, y in your environment
data = scipy.io.loadmat('ex6data3.mat')
Xval = data['Xval']
yval = data['yval'].flatten()

def dataset3Params(X, y, Xval, yval):
    c=1
    sigma=0.1
    c_choice=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    error=1
    sigma_choice=[0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    for i in range(len(c_choice)):
        for j in range(len(sigma_choice)):
            clf=svm.SVC(C=c_choice[i],kernel="rbf",gamma=1/(2*sigma_choice[j]**2))
            model=clf.fit(X,y)
            pre=model.predict(Xval)
            err=sum(pre!=yval)/np.size(yval,0)
            if err<error:
                error=err
                c=c_choice[i]
                sigma=sigma_choice[j]
                print("this time c is ",c)
                print("this time sigma is ", sigma)
    return c,sigma


# Try different SVM Parameters here
C, sigma = dataset3Params(X, y, Xval, yval)
gamma = 1.0 / (2.0 * sigma ** 2)
# Train the SVM

clf = svm.SVC(C=C, kernel='rbf', tol=1e-3, max_iter=200, gamma=gamma)
model = clf.fit(X, y)
#visualizeBoundary(X, y, model)

_=input("Program paused. Press Enter to continue...")

