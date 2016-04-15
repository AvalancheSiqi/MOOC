import numpy as np
from pylab import scatter, show, figure, subplot

# Machine Learning by Andrew Ng on Coursera week 2
# Logistic Regression


def main():
    # Load the dataset
    dataset = np.loadtxt('datasets/ex2data1.txt', delimiter=',')
    (m, n) = dataset.shape
    X = dataset[:, :n-1].reshape(m, n-1)
    y = dataset[:, n-1].reshape(m, 1)

    # Prepare the canvas
    fig = figure(figsize=(14, 6), dpi=100)
    p1 = fig.add_subplot(121)
    p2 = fig.add_subplot(122)

    # Plot all points
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    pos_points = p1.scatter(X[pos[0]].T[0], X[pos[0]].T[1], marker='o', c='y')
    neg_points = p1.scatter(X[neg[0]].T[0], X[neg[0]].T[1], marker='x', c='k')
    p1.set_xlabel('Exam1 score')
    p1.set_ylabel('Exam2 score')
    p1.legend([pos_points, neg_points], ['Admitted', 'Not Admitted'])

    # Add ones to its left
    X = np.hstack((np.ones(shape=(m, 1)), X))
    theta = np.zeros(shape=(n, 1))
    # Parameters for gradient descent
    iteration = 500
    alpha = 0.01
    theta, J_history = gradient_descent(X, y, theta, alpha, iteration)
    print J_history[-1]

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


if __name__ == '__main__':
    main()
