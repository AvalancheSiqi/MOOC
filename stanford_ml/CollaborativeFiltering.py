#!/usr/bin/env python
# encoding: utf-8

import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# Machine Learning by Andrew Ng on Coursera week 8
# Implement Collaborative Filtering


def main():
    # ===================== Part 1: Loading movie ratings dataset =====================
    print 'Loading movie ratings dataset.'
    dataset = sio.loadmat('datasets/ex8_movies.mat')
    r = dataset['R']
    y = dataset['Y']
    print 'Average rating for movie 1 (Toy Story): %f / 5\n' % np.mean(y[0, np.where(r[0, :] == 1)[0]])
    plt.imshow(y, extent=[0, 943, 0, 1682])
    plt.xlabel('Users')
    plt.ylabel('Movies')
    plt.show()

    # ================= Part 2: Collaborative Filtering Cost Function =================
    plt.close()
    dataset = sio.loadmat('datasets/ex8_movieParams.mat')
    theta = dataset['Theta']
    x = dataset['X']
    # Reduce the data set size so that this runs faster
    num_users = 4
    num_movies = 5
    num_features = 3
    x = x[:num_movies, :num_features]
    theta = theta[:num_users, :num_features]
    y = y[:num_movies, :num_users]
    r = r[:num_movies, :num_users]

    j = cofi_cost_func(np.hstack((x.flatten(), theta.flatten())), y, r, num_users, num_movies, num_features, 0)
    print 'Cost at loaded parameters: %.2f \n(this value should be about 22.22)\n' % j

    # ================= Part 3: Collaborative Filtering Gradient =================
    print 'Checking Gradients (without regularization) ... '
    check_cost_func()
    raw_input("Program paused. Press enter to continue.\n")

    # ============= Part 4: Collaborative Filtering Cost Regularization =============
    j = cofi_cost_func(np.hstack((x.flatten(), theta.flatten())), y, r, num_users, num_movies, num_features, 1.5)
    print 'Cost at loaded parameters (lambda = 1.5): %.2f \n(this value should be about 31.34)\n' % j

    # =========== Part 5: Collaborative Filtering Gradient Regularization ===========
    print 'Checking Gradients (with regularization) ... '
    check_cost_func(1.5)
    raw_input("Program paused. Press enter to continue.\n")

    # ================== Part 6: Entering ratings for a new user ==================
    movie_list = load_movie_list('datasets/movie_ids.txt')
    m = len(movie_list)
    my_ratings = np.zeros(shape=(m, 1))
    # Set some ratings
    my_ratings[1] = 4
    my_ratings[98] = 2
    my_ratings[7] = 3
    my_ratings[12] = 5
    my_ratings[54] = 4
    my_ratings[64] = 5
    my_ratings[66] = 3
    my_ratings[69] = 5
    my_ratings[183] = 4
    my_ratings[226] = 5
    my_ratings[355] = 5
    print 'New user ratings:'
    for i in range(m):
        if my_ratings[i] > 0:
            print 'Rated %d for %s' % (my_ratings[i], movie_list[i])

    # ===================== Part 7: Learning Movie Ratings =====================
    print '\nTraining collaborative filtering...'
    dataset = sio.loadmat('datasets/ex8_movies.mat')
    r = dataset['R']
    y = dataset['Y']
    y = np.hstack((my_ratings, y))
    r = np.hstack((my_ratings != 0, r))

    y_mean, y_norm = normalize_ratings(y, r)
    num_movies, num_users = y.shape
    num_features = 10

    initial_x = np.random.randn(num_movies, num_features)
    initial_theta = np.random.randn(num_users, num_features)
    initial_params = np.hstack((initial_x.flatten(), initial_theta.flatten()))

    lam = 10
    params = optimize.fmin_cg(cofi_cost_func, initial_params, fprime=cofi_grad,
                              args=(y, r, num_users, num_movies, num_features, lam), maxiter=100)
    x = params[:num_movies*num_features].reshape(num_movies, num_features)
    theta = params[num_movies*num_features:].reshape(num_users, num_features)
    print 'Recommender system learning completed.\n'

    # ===================== Part 8: Recommendation for you =====================
    p = x.dot(theta.T)
    my_predictions = p[:, 0].reshape(m, 1)+y_mean
    idx = np.argsort(my_predictions, 0).ravel()[::-1]
    print 'Top recommendations for you:'
    for i in range(10):
        j = idx[i]
        print 'Predicting rating %.1f for movie %s' % (my_predictions[j], movie_list[j])

    return


def load_movie_list(path):
    """
    Load movie list with id as key, name as value
    """
    vl = [l.rstrip().split(' ', 1) for l in open(path)]
    return {int(k): v for (k, v) in vl}


def cofi_cost_func(params, y, r, num_users, num_movies, num_features, lam):
    """
    Collaborative Filtering cost function
    :param params: concatenate of target matrix x and theta in flatten
    :param y: rating matrix with numeric value
    :param r: rating matrix with whether rated
    :param num_users: number of users
    :param num_movies: number of movies
    :param num_features: number of features
    :param lam: regularisation item
    :return: Collaborative Filtering cost function value
    """
    x = params[:num_movies*num_features].reshape(num_movies, num_features)
    theta = params[num_movies*num_features:].reshape(num_users, num_features)
    pred = np.dot(x, theta.T)
    j = (np.sum(((pred - y)*r)**2)/2 + 1.0*lam/2*np.sum(theta**2) + 1.0*lam/2*np.sum(x**2))
    return j


def cofi_grad(params, y, r, num_users, num_movies, num_features, lam):
    """
    Collaborative Filtering gradient descent
    """
    x = params[:num_movies * num_features].reshape(num_movies, num_features)
    theta = params[num_movies * num_features:].reshape(num_users, num_features)
    pred = np.dot(x, theta.T)
    x_grad = ((pred - y) * r).dot(theta) + lam * x
    theta_grad = ((pred - y) * r).T.dot(x) + lam * theta
    return np.append(x_grad.ravel(), theta_grad.ravel())


def compute_numerical_gradient(g, theta):
    """
    Computes the gradient using "finite differences" and gives us a numerical estimate of the gradient
    """
    numgrad = np.zeros(shape=theta.shape)
    perturb = np.zeros(shape=theta.shape)
    e = 1e-4
    for p in range(len(theta)):
        perturb[p] = e
        loss1 = g(theta - perturb)
        loss2 = g(theta + perturb)
        numgrad[p] = (loss2 - loss1) / (2*e)
        perturb[p] = 0
    return numgrad


def check_cost_func(lam=0):
    """
    Creates a collaborative filtering problem to check your cost function and gradients
    """
    x_t = np.random.random(size=(4, 3))
    theta_t = np.random.random(size=(5, 3))

    y = x_t.dot(theta_t.T)
    y[np.random.random(size=y.shape) > 0.5] = 0
    r = np.zeros(shape=y.shape)
    r[y != 0] = 1

    x = np.random.randn(x_t.shape[0], x_t.shape[1])
    theta = np.random.randn(theta_t.shape[0], theta_t.shape[1])
    num_movies, num_users = y.shape
    num_features = theta_t.shape[1]

    grad = cofi_grad(np.hstack((x.flatten(), theta.flatten())), y, r, num_users, num_movies, num_features, lam)
    g = lambda t: cofi_cost_func(t, y, r, num_users, num_movies, num_features, lam)
    numgrad = compute_numerical_gradient(g, np.append(x.ravel(), theta.ravel()))
    print np.hstack((grad.reshape(27, 1), numgrad.reshape(27, 1)))

    print 'The above two columns you get should be very similar.'
    print '(Left-Your Numerical Gradient, Right-Analytical Gradient)'

    diff = np.linalg.norm(numgrad-grad)/np.linalg.norm(numgrad+grad)
    print 'If your backpropagation implementation is correct, then the relative difference \n will be small' \
          '(less than 1e-9). Relative Difference: %.6e\n' % diff
    return


def normalize_ratings(y, r):
    """
    Preprocess data by subtracting mean rating for every movie (every row)
    """
    m, n = y.shape
    y_mean = np.zeros(shape=(m, 1))
    y_norm = np.zeros(shape=y.shape)
    for i in range(m):
        idx = np.where(r[i, :] == 1)[0]
        y_mean[i] = np.mean(y[i, idx])
        y_norm[i, :] = y[i, :] - y_mean[i]
    return y_mean, y_norm


if __name__ == '__main__':
    main()
