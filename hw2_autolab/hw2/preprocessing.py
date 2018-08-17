import numpy as np


def sample_zero_mean(x):
    """
    Make each sample have a mean of zero by subtracting mean along the feature axis.
    :param x: float32(shape=(samples, features))
    :return: array same shape as x
    """
    # print("x", x)
    mean = np.mean(x, axis=1, keepdims=True)
    # print("mean", mean)
    return x - mean


# print(sample_zero_mean(np.random.normal(1.0, 2.0, (3, 2))))


def gcn(x, scale=55., bias=0.01):
    """
    GCN each sample (assume sample mean=0)
    :param x: float32(shape=(samples, features))
    :param scale: factor to scale output
    :param bias: bias for sqrt
    :return: scale * x / sqrt(bias + sample variance)
    """
    # print("x", x)
    var = np.var(x, axis=1, keepdims=True)
    # print("var", var)
    return scale * x / np.sqrt(bias + var)

# print(gcn(np.random.normal(1.0, 2.0, (3, 2))))


def feature_zero_mean(x, xtest):
    """
    Make each feature have a mean of zero by subtracting mean along sample axis.
    Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features))
    :param xtest: float32(shape=(samples, features))
    :return: tuple (x, xtest)
    """
    # print("x", x)
    mean = np.mean(x, axis=0, keepdims=True)
    # print("mean", mean)
    return x - mean, xtest - mean

# print(feature_zero_mean(np.random.normal(1.0, 2.0, (3, 2)),
#                         np.random.normal(1.0, 2.0, (3, 2))
#                         ))


def zca(x, xtest, bias=0.1):
    """
    ZCA training data. Use train statistics to normalize test data.
    :param x: float32(shape=(samples, features)) (assume mean=0)
    :param xtest: float32(shape=(samples, features))
    :param bias: bias to add to covariance matrix
    :return: tuple (x, xtest)
    """
    # Calculate principal components
    # Reference: https://martin-thoma.com/zca-whitening/
    flat_x = x
    sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
    u, s, _ = np.linalg.svd(sigma)
    pc = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + 10e-7))), u.T)

    # Apply ZCA whitening
    x = np.dot(flat_x, pc)
    xtest = np.dot(xtest, pc)
    return x, xtest

# print(zca(np.random.normal(1.0, 2.0, (3, 2)),
#                         np.random.normal(1.0, 2.0, (3, 2))
#                         ))


def cifar_10_preprocess(x, xtest, image_size=32):
    """
    1) sample_zero_mean and gcn xtrain and xtest.
    2) feature_zero_mean xtrain and xtest.
    3) zca xtrain and xtest.
    4) reshape xtrain and xtest into NCHW
    :param x: float32 flat images (n, 3*image_size^2)
    :param xtest float32 flat images (n, 3*image_size^2)
    :param image_size: height and width of image
    :return: tuple (new x, new xtest), each shaped (n, 3, image_size, image_size)
    """
    x = gcn(sample_zero_mean(x))
    xtest = gcn(sample_zero_mean(xtest))
    x, xtest = feature_zero_mean(x, xtest)
    x, xtest = zca(x, xtest)
    x = x.reshape((-1, 3, image_size, image_size))
    xtest = xtest.reshape((-1, 3, image_size, image_size))
    return x, xtest
