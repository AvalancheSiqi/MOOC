import numpy as np
from pylab import show, figure, subplot, linspace
from sklearn import linear_model
from scipy import optimize

# Machine Learning by Andrew Ng on Coursera week 2
# Logistic Regression with regularisation and features mapping


def main():
    # Load the dataset
    dataset = np.loadtxt('datasets/ex2data2.txt', delimiter=',')
    (m, n) = dataset.shape
    X = dataset[:, :n-1].reshape(m, n-1)
    y = dataset[:, n-1].reshape(m, 1)

    # Prepare the canvas
    fig = figure(figsize=(14, 7), dpi=100)
    # sklearn logistic regression
    p1 = subplot(221)
    # Gradient descent by fmin_bfgs
    p2 = subplot(222)
    # Gradient descent by iteration
    p3 = subplot(223)
    # Monitor of cost function value by iteration
    p4 = subplot(224)

    pos = np.where(y == 1)
    neg = np.where(y == 0)
    # Plot all points
    p1.set_title("sklearn logistic regression")
    p2.set_title("Gradient descent by fmin_bfgs")
    p3.set_title("Gradient descent by iteration")
    for p in (p1, p2, p3):
        pos_points = p.scatter(X[pos[0]].T[0], X[pos[0]].T[1], marker='o', c='g')
        neg_points = p.scatter(X[neg[0]].T[0], X[neg[0]].T[1], marker='x', c='r')
        p.set_xlabel('Microchip Test 1')
        p.set_ylabel('Microchip Test 2')
        p.legend([pos_points, neg_points], ['Admitted', 'Not Admitted'])

    # Add ones to its left
    X = np.hstack((np.ones(shape=(m, 1)), map_feature(X)))
    n2 = X.shape[1]

    # Figure1: Leverage logistic regression model in sklearn package
    # for some reasons it doesn't plot correctly, -3 to theta_0 shift will make it more sense, unknown reason
    lr = linear_model.LogisticRegression()
    lr.fit(X, y.ravel())
    # Accuracy rate for sklearn logistic regression and plot in figure
    acc1 = np.mean(lr.predict(X) == y.ravel())*100
    print 'sklearn logistic regression train accuracy: %.4f\n' % acc1
    plot_db_points(p1, lr.coef_.ravel().reshape(n2, 1))

    # Figure2: Leverage gradient descent by fmin_ncg
    theta2 = np.ones(shape=(n2, 1))
    theta2 = optimize.fmin_ncg(compute_cost_reg, theta2, fprime=grad, args=(X, y, 1))
    # Accuracy rate for fmin_bfgs gradient descent and plot in figure
    acc2 = np.mean(predict(theta2, X) == y)*100
    print 'fmin_ncg gradient descent train accuracy: %.4f\n' % acc2
    plot_db_points(p2, theta2.reshape(n2, 1))

    # Figure3: Leverage gradient descent by iteration
    theta3 = np.ones(shape=(n2, 1))
    iteration = 50000
    alpha = 0.01
    lam = 1
    theta3, J_history = gradient_descent(X, y, theta3, alpha, iteration, lam)
    # Accuracy rate for iteration gradient descent and plot in figure
    acc3 = np.mean(predict(theta3, X) == y)*100
    print 'iteration gradient descent train accuracy: %.4f\n' % acc3
    plot_db_points(p3, theta3.reshape(n2, 1))

    # Figure4: Monitor of cost function value
    p4.set_xlabel('Iteration number')
    p4.set_ylabel('Cost function value')
    p4.plot(range(iteration), J_history)
    points = [100, 1000, 10000, 40000]
    p4.scatter(points, J_history[points], marker='o', color='k')
    for x, y in zip(points, J_history[points]):
        p4.annotate(y, xy=(x, y), ha='left', va='bottom')

    show()


def map_feature(X):
    """
    Map the features into all polynomial terms of x1 and x2 up to the sixth power
    :param X: training matrix
    :return: transformed feature matrix
    """
    m = X.shape[0]
    x1 = X[:, 0].reshape(m, 1)
    x2 = X[:, 1].reshape(m, 1)
    for k in range(1, 7):
        for i in range(0, k+1):
            j = k - i
            X = np.hstack((X, exponent(x1, i)*exponent(x2, j)))
    return X


def exponent(X, t):
    """
    Compute the t-times exponent of matrix X
    :param X: training matrix
    :param t: target exponent times
    :return: X.^t
    """
    m = X.shape[0]
    ret = np.ones([m, 1])
    for _ in range(0, t):
        ret = ret*X
    return ret


def sigmoid(z):
    """
    Map input to range zero to one
    :param z: predict value
    :return: mapped value between 0 and 1
    """
    return 1.0 / (1 + np.exp(-z))


def compute_cost_reg(theta, X, y, lam):
    """
    Compute cost function for logistic regression
    :param X: training matrix
    :param y: output target
    :param theta: hypothesis of ml model
    :return: ost function value
    """
    m = y.size
    pred = sigmoid(np.dot(X, theta))
    return -1.0/m*(np.dot(y.T, np.log(pred)) + np.dot((1-y).T, np.log(1-pred))) + lam/2.0/m*sum(theta[1:]*theta[1:])


# Gradient Descent module
def gradient_descent(X, y, theta, alpha, iteration, lam):
    """
    Conduct gradient descent for logistic regression
    :param X: training matrix
    :param y: output target
    :param theta: initial hypothesis guess
    :param alpha: learning rate
    :param iteration: iteration number
    :return: hypothesis / history list of cost function value
    """
    n = X.shape[1]
    J_history = np.zeros(shape=(iteration, 1))

    for i in range(iteration):
        theta = theta - alpha*np.array(grad(theta, X, y, lam)).reshape(n, 1)
        J_history[i, 0] = compute_cost_reg(theta, X, y, lam)

    return theta, J_history


def grad(theta, X, y, lam):
    """
    gradient for a single example
    """
    m = y.size
    theta = theta.reshape(len(theta), 1)
    pred = sigmoid(np.dot(X, theta)).reshape(m, 1)
    error = np.dot(X.T, (pred - y))
    ret = error/m + 1.0*lam/m*theta
    ret[0] -= 1.0*lam/m*theta[0]
    return ret.flatten()


# Plot decision boundary two end points
def plot_db_points(fig, theta):
    """
    Plot decision boundary two end points
    :param fig: current figure canvas
    :param X: training matrix
    :param theta: final hypothesis
    :return: two end points that exist in drawing space
    """
    # Here is the grid range
    u = linspace(-1, 1.5, 50)
    v = linspace(-1, 1.5, 50)
    z = np.zeros((len(u), len(v)))
    # Evaluate z = theta*x over the grid
    for i in range(len(u)):
        for j in range(len(v)):
            t = np.array([[u[i], v[j]]])
            z[i, j] = np.dot(np.hstack((np.ones((1, 1)), map_feature(t))), theta)
    z = z.T
    fig.contour(u, v, z, [0])
    return


def predict(theta, X):
    """
    Predict whether labeled as 0 or 1 regarding threshold
    :param theta: hypothesis
    :param X: data instance
    :return: 0 as negative while 1 as positive
    """
    return sigmoid(np.dot(X, theta.reshape(X.shape[1], 1))) >= 0.5


if __name__ == '__main__':
    main()