#!/usr/bin/env python
# encoding: utf-8

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt, floor, ceil

# Machine Learning by Andrew Ng on Coursera week 7
# Implement Principal Component Analysis


def main():
    # ==================== Part 1: Load Example Dataset ====================
    print 'Visualizing example dataset for PCA.\n'
    dataset = sio.loadmat('datasets/ex7data1.mat')
    x = dataset['X']
    m, n = x.shape
    plt.figure(figsize=(7, 7))

    # ==================== Part 2: Principal Component Analysis ====================
    print 'Running PCA on example dataset.'
    # Before running PCA, it is important to first normalize X
    x_norm, mu, sigma = feature_normalize(x)
    u, s = pca(x_norm)
    plt.xlim(0.5, 6.5)
    plt.ylim(2, 8)
    plt.scatter(x[:, 0], x[:, 1], marker='o', color='b')
    draw_line(mu, mu+1.5*s[0]*u[:, 0].T, 'k')
    draw_line(mu, mu+1.5*s[1]*u[:, 1].T, 'k')
    plt.show()
    print 'Top eigenvector: '
    print ' u(:, 0) = %f %f ' % (u[0, 0], u[1, 0])
    print '(you should expect to see -0.707107 -0.707107)\n'

    # ==================== Part 3: Dimension Reduction ====================
    plt.close()
    plt.figure(figsize=(7, 7))
    plt.xlim(-4, 3)
    plt.ylim(-4, 3)
    print 'Dimension reduction on example dataset.'
    k = 1
    z = project_data(x_norm, u, k)
    print 'Projection of the first example: %f' % z[0]
    print '(this value should be about 1.481274)'
    x_rec = recover_data(z, u, k)
    print 'Approximation of the first example: %f %f' % (x_rec[0, 0], x_rec[0, 1])
    print '(this value should be about  -1.047419 -1.047419)\n'
    for i in range(m):
        draw_line(x_norm[i, :], x_rec[i, :], c='k')
    plt.scatter(x_norm[:, 0], x_norm[:, 1], marker='o', color='b')
    plt.scatter(x_rec[:, 0], x_rec[:, 1], marker='o', color='r')
    plt.show()

    # ============== Part 4: Loading and Visualizing Face Data ==============
    plt.close()
    print 'Loading face dataset.\n'
    facedata = sio.loadmat('datasets/ex7faces.mat')['X']
    display_data(facedata[:100, :])
    plt.show()

    # ================ Part 5: PCA on Face Data: Eigenfaces ================
    plt.close()
    print 'Running PCA on face dataset. This might take a minute or two ...\n'
    x_norm, mu, sigma = feature_normalize(facedata)
    u, s = pca(x_norm)
    display_data(u[:, :36].T)
    plt.show()

    # ================ Part 6: Dimension Reduction for Faces ================
    print 'Dimension reduction for face dataset.'
    k = 100
    z = project_data(x_norm, u, k)
    print 'The projected data Z has a size of: ', z.shape, '\n'

    # ==== Part 7: Visualization of Faces after PCA Dimension Reduction ====
    print 'Visualizing the projected (reduced dimension) faces.\n'
    k = 100
    x_rec = recover_data(z, u, k)
    plt.close()
    p1 = plt.subplot(121)
    display_data(x_norm[:100, :])
    p1.set_title('Original faces')
    p2 = plt.subplot(122)
    display_data(x_rec[:100, :])
    p2.set_title('Recovered faces')
    plt.show()


def feature_normalize(x):
    """
    Conduct normalisation on training matrix which leads to a zero mean value and normal distribution
    """
    mu = np.mean(x, axis=0)
    x_norm = x - mu
    sigma = np.std(x_norm, axis=0, ddof=1)
    x_norm = x_norm/sigma
    return x_norm, mu, sigma


def pca(x):
    """
    Conduct PCA on target matrix x
    """
    m = x.shape[0]
    sigma = x.T.dot(x)/m
    u, s, _ = np.linalg.svd(sigma)
    return u, s


def project_data(x, u, k):
    """
    Project x into new space with top k vectors in u
    """
    u_reduce = u[:, :k]
    return x.dot(u_reduce)


def recover_data(z, u, k):
    """
    Recover matrix z back to x with top k vectors in u
    """
    u_reduce = u[:, :k]
    return z.dot(u_reduce.T)


def draw_line(p1, p2, c):
    """
    Draw line between previous centroids and current centroids
    """
    plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=c)


def display_data(x, example_width=None):
    """
    Displays 2D data stored in X in a nice grid
    :param x: Hand writing numbers array presentation
    :param example_width: plot example width
    :return: figure with 2D hand writing numbers data plotted
    """
    if not example_width:
        example_width = round(sqrt(x.shape[1]))

    m, n = x.shape
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(floor(sqrt(m)))
    display_cols = int(ceil(m/display_rows))

    # Between images padding
    pad = 1

    # Set up blank figure to display, with size satisfies
    display_array = -1*np.ones((pad+display_rows*(example_height+pad), pad+display_cols*(example_width+pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for i in range(display_rows):
        for j in range(display_cols):
            if curr_ex >= m:
                break
            max_val = max(abs(x[curr_ex]))
            top = pad+j*(example_height+pad)
            left = pad+i*(example_width+pad)
            display_array[top:top+example_height, left:left+example_width] \
                = x[curr_ex].reshape(example_height, example_width)/max_val
            curr_ex += 1

    # Plot the image
    plt.imshow(display_array.T, cmap="Greys_r")
    return


if __name__ == '__main__':
    main()
