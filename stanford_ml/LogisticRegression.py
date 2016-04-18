import numpy as np
from pylab import show, figure
from sklearn import linear_model

# Machine Learning by Andrew Ng on Coursera week 2
# Logistic Regression


def main():
    # Load the dataset
    dataset = np.loadtxt('datasets/ex2data1.txt', delimiter=',')
    (m, n) = dataset.shape
    X = dataset[:, :n-1].reshape(m, n-1)
    y = dataset[:, n-1].reshape(m, 1)

    # Prepare the canvas
    # fig = figure(figsize=(14, 6), dpi=100)
    # p1 = fig.add_subplot(121)
    # p2 = fig.add_subplot(122)

    # Plot all points
    # pos = np.where(y == 1)
    # neg = np.where(y == 0)
    # pos_points = p1.scatter(X[pos[0]].T[0], X[pos[0]].T[1], marker='o', c='y')
    # neg_points = p1.scatter(X[neg[0]].T[0], X[neg[0]].T[1], marker='x', c='k')
    # p1.set_xlabel('Exam1 score')
    # p1.set_ylabel('Exam2 score')
    # p1.legend([pos_points, neg_points], ['Admitted', 'Not Admitted'])

    # Add ones to its left
    X = np.hstack((np.ones(shape=(m, 1)), X))
    theta = np.zeros(shape=(n, 1))

    # Parameters for gradient descent
    iteration = 10000000
    alpha = 0.001
    theta, J_history = gradient_descent(X, y, theta, alpha, iteration)
    print 'Cost at initial theta (zeros): %f'%J_history[0]
    print 'Gradient at final theta:'
    print theta

    # Figure2: Leverage logistic regression model in sklearn package
    # lr = linear_model.LogisticRegression()
    # lr.fit(X, y.ravel())
    print 'For a student with scores 45 and 85, we predict an admission probability of %f' % sigmoid(np.dot([[1, 45, 85]], theta))
    print 'For a student with scores 15 and 45, we predict an admission probability of %f' % sigmoid(np.dot([[1, 15, 45]], theta))
    print 'For a student with scores 60 and 60, we predict an admission probability of %f' % sigmoid(np.dot([[1, 60, 60]], theta))

    acc = (((sigmoid(np.dot(X, theta))>=0.5).astype(int) == y.ravel()).astype(int).mean()*100)
    print 'Train Accuracy: %.4f\n' % acc

    # a, b = plot_db_points(X, theta)
    # p1.plot(a, b)

    # Figure2: Monitor of cost function value
    # p2.set_title('Cost function value history')
    # p2.set_xlabel('Iteration number')
    # p2.set_ylabel('Cost function value')
    # p2.plot(range(iteration), J_history)
    # points = [100, 1000, 10000, 100000, 1000000, 5000000, 8000000]
    # p2.scatter(points, J_history[points], marker='o', color='k')
    # for x, y in zip(points, J_history[points]):
    #     p2.annotate(y, xy=(x, y), ha='left', va='bottom')

    # show()


def sigmoid(X):
    '''
    Map input to range zero to one
    :param X: predict value
    :return: mapped value between 0 and 1
    '''
    return 1.0 / (1 + np.exp(-X))


def compute_cost(X, y, theta):
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
    m = y.size
    J_history = np.zeros(shape=(iteration, 1))

    for i in range(iteration):
        pred = sigmoid(np.dot(X, theta))
        errors = np.dot(X.T, (pred - y))
        theta = theta - 1.0*alpha/m*errors
        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history


# Plot decision boundary two end points
def plot_db_points(X, theta):
    '''
    Plot decision boundary two end points
    :param X: training matrix
    :param theta: final hypothesis
    :return: two end points that exist in drawing space
    '''
    if X.shape[1] <= 3:
        plot_x = [min(X[:, 2])-2, max(X[:, 2])+2]
        plot_y = (-1.0 / theta[2]) * (theta[1]*plot_x+theta[0])
    return plot_x, plot_y


if __name__ == '__main__':
    main()