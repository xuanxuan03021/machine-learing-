## Machine Learning Online Class
#  Exercise 7 | Principle Component Analysis and K-Means Clustering
#
#  Instructions
#  ------------
#
#  This file contains code that helps you get started on the
#  exercise. You will need to complete the following functions:
#
#     pca.m
#     projectData.m
#     recoverData.m
#     computeCentroids.m
#     findClosestCentroids.m
#     kMeansInitCentroids.m
#
#  For this exercise, you will not need to change any code in this file,
#  or any other files other than those mentioned above.
#

## ================= Part 1: Find Closest Centroids ====================
#  To help you implement K-Means, we have divided the learning algorithm
#  into two functions -- findClosestCentroids and computeCentroids. In this
#  part, you shoudl complete the code in the findClosestCentroids function.
#
from matplotlib import use, cm
use('TkAgg')
import numpy as np
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolor


print ('Finding closest centroids.')

# Load an example dataset that we will be using
data = scipy.io.loadmat('ex7data2.mat')
X = data['X']
print(X.shape)
print("****")
# Select an initial set of centroids
K = 3 # 3 Centroids
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Find the closest centroids for the examples using the
# initial_centroids
def findClosestCentroids(X, initial_centroids):
    distance=np.zeros((X.shape[0],K))
    for i in range(K):
        Xtemp=X-initial_centroids[i]
        dis=Xtemp**2
        ##np.power(Xtemp,2)
        distance[:,i]=np.sum(dis,axis=1)

    centroids=np.argmin(distance,axis=1)
    return centroids

idx=findClosestCentroids(X, initial_centroids)

print ('Closest centroids for the first 3 examples:')
print (idx[0:3])
print ('(the closest centroids should be 0, 2, 1 respectively)')

_=input("Program paused. Press Enter to continue...")

# ## ===================== Part 2: Compute Means =========================
# #  After implementing the closest centroids function, you should now
# #  complete the computeCentroids function.
# #
print ('Computing centroids means.')

def computeCentroids(X,idx,K):
    n=X.shape[1]
    centroids=np.zeros((K,n))
    for i in range(K):
        tempcat=np.where(idx==i)
        sum=np.sum(X[tempcat],axis=0)
        centroids[i,:]=sum/np.size(X[tempcat],0)

    return centroids


#  Compute means based on the closest centroids found in the previous part.
centroids = computeCentroids(X, idx, K)

print ('Centroids computed after initial finding of closest centroids:')
for c in centroids:
    print (c)

print ('(the centroids should be')
print( '   [ 2.428301 3.157924 ]')
print ('   [ 5.813503 2.633656 ]')
print ('   [ 7.119387 3.616684 ]')

_=input("Program paused. Press Enter to continue...")


# # ## =================== Part 3: K-Means Clustering ======================
# # #  After you have completed the two functions computeCentroids and
# # #  findClosestCentroids, you have all the necessary pieces to run the
# # #  kMeans algorithm. In this part, you will run the K-Means algorithm on
# # #  the example dataset we have provided.
# # #
# 中心点连线
def drawLine(p1, p2):
    x = np.array([p1[0], p2[0]])
    y = np.array([p1[1], p2[1]])
    plt.plot(x, y,color='black')

# 绘制数据点
def plotDataPoints(x, idx, k):
    colors = ['red', 'green', 'blue']
    #按照对应的数字划颜色
    plt.scatter(x[:, 0], x[:, 1], c=idx, s=40)

# 绘制中心点
def plotProgresskMeans(x, center, previous, idx, k, i):
    plotDataPoints(x, idx, k)
    #markersize简称MS
    #markeredgewith简称mew
    plt.plot(center[:, 0], center[:, 1], marker='x', ms=10, mew=1,c='black')#如果是list是不支持一下读取一列的list indices must be integers or slices, not tuple；NUMPY数组的则可以
    for j in range(np.size(center, 0)):
        drawLine(center[j, :], previous[j, :])
    plt.title('Iteration number %d' % (i+1))


print ('Running K-Means clustering on example dataset.')

# Load an example dataset
data = scipy.io.loadmat('ex7data2.mat')
X = data['X']

# Settings for running K-Means
K = 3
max_iters = 10

# For consistency, here we set centroids to specific values
# but in practice you want to generate them automatically, such as by
# settings them to be random examples (as can be seen in
# kMeansInitCentroids).
initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])

# Run K-Means algorithm. The 'true' at the end tells our function to plot
# the progress of K-Means
def runkMeans(X, initial_centroids, max_iters, m=False):
    previous_center=initial_centroids
    center=initial_centroids
    if m:
        plt.ion()#开启动态交互模式
        fig = plt.figure()#开始新图片
    for i in range(max_iters):
        print("K-means iteration %d/%d"%(i,max_iters))
        idx=findClosestCentroids(X,center)
        if m:
            plotProgresskMeans(X, center, previous_center, idx, K, i)
            previous_center = center

            plt.pause(0.5)#需要设定一个间歇时间才可以动态的跑起来
        center = computeCentroids(X, idx, K)
    plt.ioff()#这个要写在循环外且要写在show的前面！
    plt.show()


    return center, idx



centroids, idx = runkMeans(X, initial_centroids, max_iters, True)
print ('K-Means Done.')

_=input("Program paused. Press Enter to continue...")

## ============= Part 4: K-Means Clustering on Pixels ===============
#  In this exercise, you will use K-Means to compress an image. To do this,
#  you will first run K-Means on the colors of the pixels in the image and
#  then you will map each pixel on to it's closest centroid.
#
#  You should now complete the code in kMeansInitCentroids.m
#

print ('Running K-Means clustering on pixels from an image.')

#  Load an image of a bird
A = plt.imread('bird_small.png')

# If imread does not work for you, you can try instead
#   load ('bird_small.mat')

#A = A / 255.0 # Divide by 255 so that all values are in the range 0 - 1

# Size of the image
img_size = A.shape

# Reshape the image into an Nx3 matrix where N = number of pixels.
# Each row will contain the Red, Green and Blue pixel values
# This gives us our dataset matrix X that we will use K-Means on.
X = A.reshape(img_size[0] * img_size[1], 3)
print(X.shape)

# Run your K-Means algorithm on this data
# You should try different values of K and max_iters here
K = 16
max_iters = 10

def kMeansInitCentroids(X,K):
    m,n=X.shape
    print(m,n)
    initialCentroidNum=np.random.randint(0,np.size(X,0),size=K)
    print(initialCentroidNum)
    initial_centroids=X[initialCentroidNum,:]
    print(initial_centroids)
    return initial_centroids


# When using K-Means, it is important the initialize the centroids
# randomly.
# You should complete the code in kMeansInitCentroids.m before proceeding
initial_centroids = kMeansInitCentroids(X, K)

# Run K-Means
centroids, idx = runkMeans(X, initial_centroids, max_iters)

_=input("Program paused. Press Enter to continue...")


## ================= Part 5: Image Compression ======================
#  In this part of the exercise, you will use the clusters of K-Means to
#  compress an image. To do this, we first find the closest clusters for
#  each example. After that, we

print ('Applying K-Means to compress an image.')

# Find closest cluster members
idx = findClosestCentroids(X, centroids)

# Essentially, now we have represented the image X as in terms of the
# indices in idx.

# We can now recover the image from the indices (idx) by mapping each pixel
# (specified by it's index in idx) to the centroid value
X_recovered = np.array([centroids[e] for e in idx])

# Reshape the recovered image into proper dimensions
X_recovered = X_recovered.reshape(img_size[0], img_size[1], 3)

# Display the original image
plt.subplot(1, 2, 1)
plt.imshow(A)
plt.title('Original')
# Display compressed image side by side
plt.subplot(1, 2, 2)
plt.imshow(X_recovered)#知处理图像不显示
plt.title('Compressed, with %d colors.' % K)
plt.show()

_=input("Program paused. Press Enter to continue...")