from PIL import Image
import numpy as np
# load and display an image with Matplotlib
from matplotlib import image
from matplotlib import pyplot
import time


# calculate f(e^f) in order to use logsumexp trick and avoid underflow/overflow
def f_logsumexp(x, mean, var, pi):
    N = x.shape[0]
    K = mean.shape[0]
    trick = np.zeros((N, K))
    for k in range(K):
        subtraction = np.subtract(x, mean[k])
        arg1 = -1.0 / (2 * var[k]) * np.power(subtraction, 2)
        arg2 = -0.5*np.log(2*np.pi*var[k])
        arg3 = np.sum(arg2 + arg1, axis=1)  # before sum 1xD -> 1x1
        arithmitis = np.log(pi[k]) + arg3
        trick[:, k] = arithmitis
    # find max of all fk(trick[k]) for each example
    m = trick.max(axis=1)  # Nx1
    m = m.reshape((m.shape[0], 1))
    return trick, m


# N -> number of examples
# K -> number of clusters
def update_gamma(f, m):
    f = f-m
    f = np.exp(f)   # NxK
    par = np.sum(f, axis=1)  # Nx1
    par = par.reshape((par.shape[0],1))
    result = np.divide(f, par)   # NxK
    return result


# return matrix with dimensions KxD
def update_mean(gamma, x):
    arith = np.dot(np.transpose(gamma), x)   # (KxN)*(NxD)-> KxD
    paran = np.sum(gamma, axis=0)  # Kx1
    paran = paran.reshape((paran.shape[0], 1))
    result = arith/paran    # KxD
    return result


# return vector with dimensions 1xK
def update_variance(gamma, x, mean):
    D = x.shape[1]
    K = mean.shape[0]
    arith = np.zeros((K, 1))
    for k in range(K):
        gamma_k = gamma[:, k]
        gamma_k = gamma_k.reshape((gamma_k.shape[0], 1))
        subtraction = np.subtract(x, mean[k])   # NxD
        # ((Nx1).*(NxD)-> NxD->sum row wise -> 1xN -> sum -> 1x1
        sub = np.sum(np.sum(np.multiply(np.power(subtraction, 2), gamma_k), axis=1))
        arith[k] = sub
    paran = D * np.sum(gamma, axis=0)   # Kx1
    paran = paran.reshape((K, 1))  # Kx1
    return arith/paran


def update_loglikehood(f, m):
    f = f - m   # NxK
    arg1 = np.sum(np.exp(f), axis=1)  # Nx1
    arg1 = np.log(arg1)   # Nx1
    arg1 = arg1.reshape((arg1.shape[0], 1))
    arg2 = arg1+m
    return np.sum(arg2, axis=0)  # 1x1


def init_parameters(D, K):
    mean = np.random.rand(K, D)
    var = np.random.uniform(low=0.1, high=1, size=K)    # Kx1
    val = 1/K
    pi = np.full(K, val)  # Kx1
    return mean, var, pi


# pi is not np.pi = 3.14.... is a different variable
def EM(x, K, tol):
    # counter in order to count iterations and stop after some in order our program doesn't run for an eternity
    counter = 1
    # num of examples(Here pixels)
    N = x.shape[0]
    # num of dimensions of each examples(Here RGB canals)
    D = x.shape[1]
    # init parameters
    mean, var, pi = init_parameters(D, K)
    # logsumexp trick
    f, m = f_logsumexp(x, mean, var, pi)
    loglikehood = update_loglikehood(f, m)
    while counter <= 400:
        print('Iteration: ', counter)
        # E-step
        gamma = update_gamma(f, m)  # NxK
        # M-step
        # update pi
        pi = (np.sum(gamma, axis=0))/N
        # update mean
        mean = update_mean(gamma, x)
        # update variance(var)
        var = update_variance(gamma, x, mean)
        old_loglikehood = loglikehood
        # logsumexp trick
        f, m = f_logsumexp(x, mean, var, pi)
        loglikehood = update_loglikehood(f, m)
        # check if algorithm is correct
        if loglikehood-old_loglikehood < 0:
            print('Error found in EM algorithm')
            print('Number of iterations: ', counter)
            exit()
        # check if the convergence criterion is met
        if abs(loglikehood-old_loglikehood) < tol:
            print('Convergence criterion is met')
            print('Total iterations: ', counter)
            return mean, gamma
        # update 'safety valve' in order to not loop for an eternity
        counter += 1
    return mean, gamma


def error_reconstruction(x, means_of_data):
    N = x.shape[0]
    x = x*255
    x = x.astype(np.uint8)
    diff = x-means_of_data
    sum1 = np.sqrt(np.sum(np.power(diff, 2)))
    error = sum1/N
    return error


def reconstruct_image(x, mean, gamma, K):
    D = mean.shape[1]
    # denormalize values
    mean = mean * 255
    # set data-type uint8 so every data is in set [0,255]
    mean = mean.astype(np.uint8)
    max_likelihood = np.argmax(gamma, axis=1)  # 1xN
    # matrix that has for each example(pixel) the means of dimensions(R,G,B) of k(=cluster) with highest
    # a  posteriori probability gamma. This matrix is our new data(pixels)
    means_of_data = np.array([mean[i] for i in max_likelihood])  # NxD
    # set data-type uint8 so every data is in set [0,255]
    means_of_data = means_of_data.astype(np.uint8)
    # calculate error
    error = error_reconstruction(x, means_of_data)
    print('Error of reconstruction:', error)
    means_of_data = means_of_data.reshape((height, width, D))
    segmented_image = Image.fromarray(means_of_data, mode='RGB')
    name = 'Segmented_Images\segmented_image_'+str(K)+'.jpg'
    segmented_image.save(name)


def run(x, cluster, tol):
    for K in cluster:
        print('------ Cluster: '+str(K)+' ------')
        start_time = time.time()
        mean, gamma = EM(x, K, tol)
        end_time = time.time()
        em_time = end_time-start_time
        print("Time of execution of EM for clusters/k = %s  is %s seconds " % (K, em_time))
        reconstruct_image(x, mean, gamma, K)


tolerance = 1e-6
clusters = [1, 2, 4, 8, 16, 32, 64]
path = 'Image\im.jpg'
# load image as pixel array
data = image.imread(path)
data = np.asarray(data)
# summarize shape of the pixel array
print("Dimensions of image: ", data.shape)
(height, width, d) = data.shape
max_value = np.amax(data)
# display the array of pixels as an image
pyplot.imshow(data)
pyplot.show()
# N = number of data set (Here height*width of image)
# D = dimensions of each data (Here R,G,B)
dataset = data.reshape((height*width, d))    # NxD
# normalize data
dataset = dataset/max_value
run(dataset, clusters, tolerance)
