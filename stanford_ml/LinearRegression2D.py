import numpy as np
from pylab import show, figure, annotate
from sklearn import linear_model
from mpl_toolkits.mplot3d import Axes3D

# Machine Learning by Andrew Ng on Coursera week 1
# Linear Regression with Multiple variable (2D)


def main():
    # Load the dataset
    dataset = np.loadtxt('datasets/ex1data2.txt', delimiter=',')
    m = dataset.shape[0]
    X = dataset[:, :2].reshape(m, 2)
    y = dataset[:, 2].reshape(m, 1)

    # Create 3D axes
    fig = figure(figsize=(14, 6), dpi=100)
    p1 = fig.add_subplot(121, projection='3d')
    p2 = fig.add_subplot(122)

    xs = X[:, 0]/100
    ys = X[:, 1]
    zs = y/10000
    p1.scatter(xs, ys, zs, c='b', marker='o')

    p1.set_xlabel('Size of the house in 100s')
    p1.set_ylabel('Number of bedrooms')
    p1.set_zlabel('Price of the house in $10,000s')

    # Normalise raw dataset, add ones to its left
    X = feature_normalise(X)
    X = np.hstack((np.ones(shape=(m, 1)), X))
    theta = np.zeros(shape=(3, 1))

    # Parameters for gradient descent
    iteration = 1500
    alpha = 0.01
    theta, J_history = gradient_descent(X, y, theta, alpha, iteration)

    # Plot the 3D line
    [a, b] = xs[: 2]
    [c, d] = ys[: 2]
    p1.plot(xs, ys, np.dot(X, theta).flatten())

    # Figure2: Monitor of cost function value
    p2.set_title('Cost function value history')
    p2.set_xlabel('Iteration number')
    p2.set_ylabel('Cost function value')
    p2.plot(range(iteration), J_history)
    points = [10, 100, 1000]
    p2.scatter(points, J_history[points], marker='o', color='k')
    for x, y in zip(points, J_history[points]):
        annotate(y, xy=(x, y), ha='left', va='bottom')

    show()


# Feature scale to normalise raw data
def feature_normalise(X):
    '''
    Normalise the matrix
    :param X: training matrix
    :return: normalised training matrix
    '''
    (m, n) = X.shape
    for j in range(n):
        mean_val = np.mean(X[:, j])
        std_val = np.std(X[:, j])
        X[:, j] = (X[:, j] - mean_val) / std_val
    return X


# Evaluate the cost of linear regression
def compute_cost(X, y, theta):
    '''
    Compute cost of linear regression
    :param X: training matrix
    :param y: output target
    :param theta: hypothesis of ml model
    :return: cost function value
    '''
    m = y.size
    pred = np.dot(X, theta)
    sq_errors = sum((pred-y)**2)
    J = 1.0/2/m*(sq_errors)
    return J


# Gradient Descent module
def gradient_descent(X, y, theta, alpha, iteration):
    '''
    Conduct gradient descent
    :param X: training matrix
    :param y: output target
    :param theta: initial hypothesis guess
    :param alpha: learning rate
    :param iter: iteration number
    :return: hypothesis / history list of cost function value
    '''
    m = y.size
    J_history = np.zeros(shape=(iteration, 1))

    for i in range(iteration):
        pred = np.dot(X, theta)
        errors = np.dot(X.T, (pred - y))
        theta = theta - 1.0*alpha/m*errors
        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history


if __name__ == '__main__':
    main()