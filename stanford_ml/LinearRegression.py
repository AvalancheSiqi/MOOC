from numpy import loadtxt, ones, zeros, hstack, dot, linspace, logspace
from pylab import show, figure, subplot
from sklearn import linear_model

# Machine Learning by Andrew Ng on Coursera week 1
# Linear Regression with one variable


def main():
    # Load the dataset
    dataset = loadtxt('datasets/ex1data1.txt', delimiter=',')
    m = dataset.shape[0]
    X = dataset[:, 0].reshape(m, 1)
    y = dataset[:, 1].reshape(m, 1)

    # Prepare the canvas
    figure(figsize=(14, 10), dpi=100)
    # Hypothesis by gradient descent
    p1 = subplot(221)
    # Hypothesis by sklearn linear regression
    p2 = subplot(222)
    # Monitor of cost function value
    p3 = subplot(223)
    # Contour figure with respect to theta_0 and theta_1
    p4 = subplot(224)

    # Plot the scatter points
    for i in (p1, p2):
        i.scatter(X, y, marker='x', c='r')
        i.set_title('Profit distribution')
        i.set_xlabel('Population of City in 10,000s')
        i.set_ylabel('Profit in $10,000s')

    # Manipulate the dataset, add ones to its left
    X = hstack((ones(shape=(m, 1)), X))
    theta = zeros(shape=(2, 1))

    # Parameters for gradient descent
    iteration = 1500
    alpha = 0.01
    theta, J_history = gradient_descent(X, y, theta, alpha, iteration)

    # Figure3: Monitor of cost function value
    p3.set_title('Cost function value history')
    p3.set_xlabel('Iteration number')
    p3.set_ylabel('Cost function value')
    p3.plot(range(iteration), J_history)
    p3.scatter(10, J_history[10], marker='o', color='k')
    p3.scatter(100, J_history[100], marker='o', color='k')
    p3.scatter(1000, J_history[1000], marker='o', color='k')

    # Figure1: Leverage gradient descent
    pred = X.dot(theta)
    p1.plot(X[:, 1], pred)
    p1.scatter(3.5, dot([1, 3.5], theta), marker='o', color='k')
    p1.scatter(7, dot([1, 7], theta), marker='o', color='k')

    # Figure2: Leverage linear regression model in sklearn package
    lr = linear_model.LinearRegression()
    lr.fit(X, y)
    p2.plot(X[:, 1], lr.predict(X))
    p2.scatter(3.5, lr.predict([[1, 3.5]]), marker='o', color='k')
    p2.scatter(7, lr.predict([[1, 7]]), marker='o', color='k')

    # Figure4: Contour figure with respect to theta_0 and theta_1
    # Grid over which we will calculate J
    theta0_vals = linspace(-10, 10, 100)
    theta1_vals = linspace(-1, 4, 100)

    # initialize J_vals to a matrix of 0's
    J_vals = zeros(shape=(theta0_vals.size, theta1_vals.size))

    # Fill out J_vals
    for t1, element1 in enumerate(theta0_vals):
        for t2, element2 in enumerate(theta1_vals):
            J_vals[t1, t2] = compute_cost(X, y, [[element1], [element2]])

    # Contour plot
    J_vals = J_vals.T
    # Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    p4.contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
    p4.set_title('Contour, showing minimum')
    p4.set_xlabel('theta_0')
    p4.set_ylabel('theta_1')
    p4.scatter(theta[0][0], theta[1][0], marker='x', color='r')

    show()


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
    pred = dot(X, theta)
    sq_errors = sum((pred-y)**2)
    J = 1.0/2/m*(sq_errors)
    return J


# Gradient Descent module
def gradient_descent(X, y, theta, alpha, iter):
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
    J_history = zeros(shape=(iter, 1))

    for i in range(iter):
        pred = dot(X, theta)
        errors = dot(X.T, (pred - y))
        theta = theta - 1.0*alpha/m*errors
        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history


if __name__ == '__main__':
    main()