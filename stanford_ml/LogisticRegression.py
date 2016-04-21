import numpy as np
from pylab import show, figure, subplot
from sklearn import linear_model
from scipy import optimize

# Machine Learning by Andrew Ng on Coursera week 2
# Logistic Regression


def main():
    # Load the dataset
    dataset = np.loadtxt('datasets/ex2data1.txt', delimiter=',')
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
    p1.set_title("Gradient descent by fmin_bfgs")
    p1.set_title("Gradient descent by iteration")
    for p in (p1, p2, p3):
        pos_points = p.scatter(X[pos[0]].T[0], X[pos[0]].T[1], marker='o', c='y')
        neg_points = p.scatter(X[neg[0]].T[0], X[neg[0]].T[1], marker='x', c='k')
        p.set_xlabel('Exam1 score')
        p.set_ylabel('Exam2 score')
        p.legend([pos_points, neg_points], ['Admitted', 'Not Admitted'])

    # Add ones to its left
    X = np.hstack((np.ones(shape=(m, 1)), X))

    # Figure1: Leverage logistic regression model in sklearn package
    # for some reasons it doesn't plot correctly, -3 to theta_0 shift will make it more sense, unknown reason
    lr = linear_model.LogisticRegression()
    lr.fit(X, y.ravel())
    # Accuracy rate for sklearn logistic regression and plot in figure
    acc1 = (lr.predict(X) == y.ravel()).astype(int).mean()*100
    print 'sklearn logistic regression train accuracy: %.4f\n' % acc1
    plot_db_points(p1, X, lr.coef_.ravel().reshape(n, 1))

    # Figure2: Leverage gradient descent by fmin_bfgs
    theta2 = np.ones(shape=(n, 1))
    theta2 = optimize.fmin_bfgs(compute_cost, theta2, fprime=grad, args=(X, y))
    # Accuracy rate for fmin_bfgs gradient descent and plot in figure
    acc2 = (((sigmoid(np.dot(X, theta2)) >= 0.5).astype(int) == y.ravel()).astype(int).mean()*100)
    print 'fmin_bfgs gradient descent train accuracy: %.4f\n' % acc2
    plot_db_points(p2, X, theta2.reshape(n, 1))

    # Figure3: Leverage gradient descent by iteration
    theta3 = np.ones(shape=(n, 1))
    iteration = 1000000
    alpha = 0.001
    theta3, J_history = gradient_descent(X, y, theta3, alpha, iteration)
    # Accuracy rate for iteration gradient descent and plot in figure
    acc3 = (((sigmoid(np.dot(X, theta3))>=0.5).astype(int) == y.ravel()).astype(int).mean()*100)
    print 'iteration gradient descent train accuracy: %.4f\n' % acc3
    plot_db_points(p3, X, theta3.reshape(n, 1))

    # Figure4: Monitor of cost function value
    p4.set_xlabel('Iteration number')
    p4.set_ylabel('Cost function value')
    p4.set_xlim([-100000, 1100000])
    p4.plot(range(iteration), J_history)
    points = [1000, 100000, 900000]
    p4.scatter(points, J_history[points], marker='o', color='k')
    for x, y in zip(points, J_history[points]):
        p4.annotate(y, xy=(x, y), ha='left', va='bottom')

    show()


def sigmoid(X):
    '''
    Map input to range zero to one
    :param X: predict value
    :return: mapped value between 0 and 1
    '''
    return 1.0 / (1 + np.exp(-X))


def compute_cost(theta, X, y):
    '''
    Compute cost function for logistic regression
    :param X: training matrix
    :param y: output target
    :param theta: hypothesis of ml model
    :return: ost function value
    '''
    m = y.size
    pred = sigmoid(np.dot(X, theta))
    return -1.0/m*(np.dot(y.T, np.log(pred)) + np.dot((1-y).T, np.log(1-pred)))


# Gradient Descent module
def gradient_descent(X, y, theta, alpha, iteration):
    '''
    Conduct gradient descent for logistic regression
    :param X: training matrix
    :param y: output target
    :param theta: initial hypothesis guess
    :param alpha: learning rate
    :param iteration: iteration number
    :return: hypothesis / history list of cost function value
    '''
    n = X.shape[1]
    J_history = np.zeros(shape=(iteration, 1))

    for i in range(iteration):
        theta = theta - alpha*np.array(grad(theta, X, y)).reshape(n, 1)
        J_history[i, 0] = compute_cost(theta, X, y)

    return theta, J_history


def grad(theta, X, y):
    '''
    gradient for a single example
    '''
    m = y.size
    pred = sigmoid(np.dot(X, theta)).reshape(m, 1)
    error = np.dot(X.T, (pred - y))
    return (error / m).flatten()


# Plot decision boundary two end points
def plot_db_points(fig, X, theta):
    '''
    Plot decision boundary two end points
    :param fig: current figure canvas
    :param X: training matrix
    :param theta: final hypothesis
    :return: two end points that exist in drawing space
    '''
    if X.shape[1] <= 3:
        plot_x = [min(X[:, 2]-2), max(X[:, 2]+2)]
        plot_y = (-1.0 / theta[2]) * (theta[1]*plot_x+theta[0])
        fig.plot(plot_x, plot_y)
    return


if __name__ == '__main__':
    main()