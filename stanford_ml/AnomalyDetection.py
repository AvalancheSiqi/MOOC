#!/usr/bin/env python
# encoding: utf-8

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from math import pi

# Machine Learning by Andrew Ng on Coursera week 8
# Implement Anomaly Detection


def main():
    # ===================== Part 1: Load Example Dataset =====================
    print 'Visualizing example dataset for outlier detection.\n'
    dataset = sio.loadmat('datasets/ex8data1.mat')
    x = dataset['X']
    plt.figure()
    plt.scatter(x[:, 0], x[:, 1], marker='x', color='b')
    plt.xlim(0, 30)
    plt.ylim(0, 30)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.show()

    # ================ Part 2: Estimate the dataset statistics ================
    plt.close()
    print 'Visualizing Gaussian fit.\n'
    mu, sigma2 = estimate_gaussian(x)
    p = multivariate_gaussian(x, mu, sigma2)
    visualize_fit(x, mu, sigma2)

    # ========================= Part 3: Find Outliers =========================
    xval = dataset['Xval']
    yval = dataset['yval']
    pval = multivariate_gaussian(xval, mu, sigma2)
    epsilon, f1 = select_threshold(yval, pval)
    print 'Best epsilon found using cross-validation: %.4e' % epsilon
    print 'Best F1 on Cross Validation Set:  %f' % f1
    print '   (you should see a value epsilon of about 8.99e-05)\n'

    outliers = p < epsilon
    plt.scatter(x[outliers, 0], x[outliers, 1], facecolors='none', edgecolors='r', s=50)
    plt.show()

    # ==================== Part 4: Multidimensional Outliers ====================
    plt.close()
    dataset = sio.loadmat('datasets/ex8data2.mat')
    x = dataset['X']
    xval = dataset['Xval']
    yval = dataset['yval']
    mu, sigma2 = estimate_gaussian(x)
    p = multivariate_gaussian(x, mu, sigma2)
    pval = multivariate_gaussian(xval, mu, sigma2)
    epsilon, f1 = select_threshold(yval, pval)
    print 'Best epsilon found using cross-validation: %.4e' % epsilon
    print 'Best F1 on Cross Validation Set:  %f' % f1
    print '# Outliers found: %d' % sum(p < epsilon)
    print '   (you should see a value epsilon of about 1.38e-18)\n'


def estimate_gaussian(x):
    """
    Estimate the parameters of a Gaussian distribution using the data in x
    """
    m, n = x.shape
    mu = np.mean(x, axis=0)
    sigma2 = np.array([np.sum((x[:, i]-mu[i])**2)/m for i in range(n)])
    return mu, sigma2


def multivariate_gaussian(x, mu, sigma2):
    """
    Computes the probability density function of the multivariate gaussian distribution
    """
    k = float(len(mu))
    sigma2 = np.diag(sigma2)
    x = x - mu
    p = (2*pi)**(-k/2) * np.linalg.det(sigma2)**(-0.5) * np.exp(-0.5*np.sum((x.dot(np.linalg.pinv(sigma2))*x), axis=1))
    return p


def visualize_fit(x, mu, sigma2):
    """
    Visualize the dataset and its estimated distribution
    """
    xv, yv = np.meshgrid(np.linspace(0, 35, 71), np.linspace(0, 35, 71))
    m = len(xv.ravel())
    x1 = xv.ravel().reshape(m, 1)
    y1 = yv.ravel().reshape(m, 1)
    z = multivariate_gaussian(np.hstack((x1, y1)), mu, sigma2)
    z = z.reshape(xv.shape)
    plt.xlim(0, 35)
    plt.ylim(0, 35)
    plt.xlabel('Latency (ms)')
    plt.ylabel('Throughput (mb/s)')
    plt.scatter(x[:, 0], x[:, 1], marker='x', color='b', s=10)
    levels = 10.0**np.arange(-20, 0, 3)
    plt.contour(xv, yv, z, levels)
    return


def select_threshold(yval, pval):
    """
    Find the best threshold (epsilon) to use for selecting outliers
    """
    best_epsilon = 0
    best_f1 = 0
    for e in np.linspace(min(pval), max(pval), 1000):
        predictions = (pval < e)
        if not np.sum(predictions):
            continue
        precision = 1.0*np.sum(np.logical_and(predictions == 1, yval.ravel() == 1))/np.sum(predictions)
        recall = 1.0*np.sum(np.logical_and(predictions == 1, yval.ravel() == 1))/np.sum(yval)
        f1 = 2*precision*recall/(precision+recall)
        if f1 > best_f1:
            best_f1 = f1
            best_epsilon = e
    return best_epsilon, best_f1


if __name__ == '__main__':
    main()
