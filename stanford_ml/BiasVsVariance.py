import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
from scipy import optimize

# Machine Learning by Andrew Ng on Coursera week 5
# Implement regularized linear regression and use it to study models with different bias-variance properties


def main():
    # == == == == == = Part 1: Loading and Visualizing Data == == == == == == = #
    print "Loading and Visualizing Data ...\n"
    dataset = sio.loadmat('datasets/ex5data1.mat')
    X = dataset['X']
    y = dataset['y']
    Xval = dataset['Xval']
    yval = dataset['yval']
    Xtest = dataset['Xtest']
    ytest = dataset['ytest']
    m, n = X.shape

    fig = plt.figure()
    p1 = fig.add_subplot(111)
    p1.scatter(X, y, marker='x', color='r', linewidths=1.5)
    p1.set_xlabel('Change in water level (x)')
    p1.set_ylabel('Water flowing out of the dam (y)')

    # == == == == == == == == Part 2: Train Linear Regression == == == == == == = #
    X = np.hstack((np.ones(shape=(m, 1)), X))
    lam = 1
    theta = train_linear_reg(X, y, lam)
    theta = theta.reshape(n+1, 1)

    # Plot fit over the data
    line_points = np.array([-50, 50, 1])
    print line_points
    p1.scatter(X[:, 1:], X.dot(theta), marker='+', linewidths=2)

    plt.show()
    return


def cost(theta, X, y, lam):
    m, n = X.shape
    theta = theta.reshape(n, 1)
    return 1.0/2/m*sum((X.dot(theta) - y)**2) + lam/2.0/m*sum(theta[1:]**2)


def grad(theta, X, y, lam):
    m, n = X.shape
    theta = theta.reshape(n, 1)
    err = np.dot(X.T, X.dot(theta) - y)
    ret = err/m + 1.0*lam/m*theta
    ret[0] -= 1.0*lam/m*theta[0]
    return ret.flatten()


def train_linear_reg(X, y, lam):
    m, n = X.shape
    initial_theta = np.zeros(shape=(n, 1))
    theta = optimize.fmin_cg(cost, initial_theta, fprime=grad, args=(X, y, lam), maxiter=200)
    return theta


if __name__ == '__main__':
    main()
