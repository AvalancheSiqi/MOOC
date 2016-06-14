#!/usr/bin/env python
# encoding: utf-8

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt

# Machine Learning by Andrew Ng on Coursera week 7
# Implement K-Means Clustering


def main():
    # ==================== Part 1: Find Closest Centroids ====================
    print 'Finding closest centroids.\n'
    dataset = sio.loadmat('datasets/ex7data2.mat')
    X = dataset['X']
    K = 3
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    idx = find_closest_centroids(X, initial_centroids)
    print 'Closest centroids for the first 3 examples:', idx[:3].flatten(), ' \n'
    print '(the closest centroids should be 0, 2, 1 respectively)\n'

    # ======================== Part 2: Compute Means ========================
    print 'Computing centroids means.\n'
    centroids = compute_centroids(X, idx, K)
    print 'Centroids computed after initial finding of closest centroids: '
    print centroids
    print '\n(the centroids should be'
    print '   [ 2.428301 3.157924 ]'
    print '   [ 5.813503 2.633656 ]'
    print '   [ 7.119387 3.616684 ]\n'

    # ===================== Part 3: K-Means Clustering =====================
    raw_input("Program paused. Press enter to continue.\n")

    print 'Running K-Means clustering on example dataset.\n'
    dataset = sio.loadmat('datasets/ex7data2.mat')
    X = dataset['X']
    initial_centroids = np.array([[3, 3], [6, 2], [8, 5]])
    max_iters = 10
    centroids, idx = run_kmeans(X, initial_centroids, max_iters, plot_progress=True)
    print 'K-Means Done.\n'

    # ================ Part 4: K-Means Clustering on Pixels ================
    raw_input("Program paused. Press enter to continue.\n")
    plt.close()

    print 'Running K-Means clustering on pixels from an image.\n'
    A = plt.imread('datasets/bird_small.png')
    a, b, c = A.shape
    X = A.reshape(a*b, c)
    K = 16
    max_iters = 10
    initial_centroids = kmeans_init_centroids(X, K)
    centroids, idx = run_kmeans(X, initial_centroids, max_iters)

    # ===================== Part 5: Image Compression =====================
    print '\nApplying K-Means to compress an image.\n'
    idx = find_closest_centroids(X, centroids)
    X_recovered = centroids[idx, :]
    X_recovered = X_recovered.reshape(a, b, c)
    fig2 = plt.figure()
    p1 = fig2.add_subplot(121)
    p2 = fig2.add_subplot(122)
    p1.imshow(A)
    p1.set_title('Original')
    p2.imshow(X_recovered)
    p2.set_title('Compressed, with %d colors' % K)
    plt.show()


def kmeans_init_centroids(X, K):
    """
    Initialize K centroids from random X rows
    """
    randidx = np.random.permutation(X.shape[0])
    return X[randidx[:K], :]


def find_closest_centroids(X, centroids):
    """
    Mark each row in X with closest centroids
    """
    K = centroids.shape[0]
    m = X.shape[0]
    idx = np.zeros(shape=(m, 1), dtype=int)
    for i in range(m):
        min_dist = float('inf')
        for j in range(K):
            dist = np.sum((X[i, :]-centroids[j, :])**2)
            if dist < min_dist:
                min_dist = dist
                idx[i, :] = j
    return idx


def compute_centroids(X, idx, K):
    """
    With marked label idx, compute the new centroid values
    """
    n = X.shape[1]
    centroids = np.zeros(shape=(K, n))
    for i in range(K):
        centroids[i, :] = np.mean(X[np.where(idx == i)[0], :], 0)
    return centroids


def run_kmeans(X, initial_centroids, max_iters, plot_progress=False):
    """
    Run KMeans clustering with initial_centroids, plot progress optional
    """
    if plot_progress:
        fig1 = plt.figure()

    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(shape=(m, 1))

    for i in range(max_iters):
        print 'K-Means iteration %d/%d...' % (i+1, max_iters)
        idx = find_closest_centroids(X, centroids)
        if plot_progress:
            plot_progress_kmeans(X, centroids, previous_centroids, idx, i)
            previous_centroids = centroids
        centroids = compute_centroids(X, idx, K)

    if plot_progress:
        fig1.show()

    return centroids, idx


def plot_progress_kmeans(X, centroids, previous_centroids, idx, i):
    """
    Plot the progress of KMeans, how the centroids change
    """
    colors = "bgrcmykw"
    plot_data_points(X, idx, colors)
    for j in range(centroids.shape[0]):
        draw_line(centroids[j, :], previous_centroids[j, :], colors[j])
    plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', color='k', s=80)
    plt.title('Iteration number %d' % (i+1))
    plt.pause(1)


def plot_data_points(X, idx, colors):
    """
    Plot all data points with different colors by marked label
    """
    plt.scatter(X[:, 0], X[:, 1], color=[colors[i] for i in idx], s=5)


def draw_line(p1, p2, c):
    """
    Draw line between previous centroids and current centroids
    """
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c)


if __name__ == '__main__':
    main()
