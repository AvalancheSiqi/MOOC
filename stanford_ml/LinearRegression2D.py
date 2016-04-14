import numpy as np
from pylab import show, figure
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

    xs = X[:, 0]/100.0
    ys = X[:, 1]
    zs = y/10000.0
    p1.scatter(xs, ys, zs, c='b', marker='o')

    p1.set_xlabel('Size of the house in 100s')
    p1.set_ylabel('Number of bedrooms')
    p1.set_zlabel('Price of the house in $10,000s')

    # Normalise raw dataset, add ones to its left
    X, ms = feature_normalise(X)
    X = np.hstack((np.ones(shape=(m, 1)), X))
    theta = np.zeros(shape=(3, 1))

    # Parameters for gradient descent
    iteration = 500
    alpha = 0.01
    theta, J_history = gradient_descent(X, y, theta, alpha, iteration)

    # Predicted price of a 1650 sq-ft, 3 br house
    res = np.dot(([1, (1650-ms[0][0])/ms[0][1], (3-ms[1][0])/ms[1][1]]), theta)
    print "Predicted price of a 1650 sq-ft, 3 br house: $%f"%res[0]
    p1.scatter(1650/100.0, 3, res/10000.0, c='r', marker='x')

    # Figure2: Monitor of cost function value
    p2.set_title('Cost function value history')
    p2.set_xlabel('Iteration number')
    p2.set_ylabel('Cost function value')
    p2.plot(range(iteration), J_history)
    points = [10, 100, 300]
    p2.scatter(points, J_history[points], marker='o', color='k')
    for x, y in zip(points, J_history[points]):
        p2.annotate(y, xy=(x, y), ha='left', va='bottom')

    show()


# Feature scale to normalise raw data
def feature_normalise(X):
    '''
    Normalise the matrix
    :param X: training matrix
    :return: normalised training matrix / mean & std of each feature
    '''
    (m, n) = X.shape
    ms = np.ones(shape=(n, 2))
    for j in range(n):
        ms[j][0] = np.mean(X[:, j])
        ms[j][1] = np.std(X[:, j])
        X[:, j] = (X[:, j] - ms[j][0]) / ms[j][1]
    return X, ms


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
    J = 1.0/2/m*sq_errors
    return J


# Gradient Descent module
def gradient_descent(X, y, theta, alpha, iteration):
    '''
    Conduct gradient descent
    :param X: training matrix
    :param y: output target
    :param theta: initial hypothesis guess
    :param alpha: learning rate
    :param iteration: iteration number
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