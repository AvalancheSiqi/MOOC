from numpy import loadtxt, ones, zeros, hstack, dot, linspace, logspace
from pylab import show, figure, subplot

# Machine Learning by Andrew Ng on Coursera week 1
# Linear Regression with one variable

def main():
    # Load the dataset
    dataset = loadtxt('datasets/ex1data1.txt', delimiter=',')

    # Prepare the canvas
    figure(figsize=(13, 5), dpi=100)
    p1 = subplot(121)
    p2 = subplot(122)

    (m, n) = dataset.shape
    X = dataset[:, :n-1].reshape(m, n-1)
    y = dataset[:, n-1].reshape(m, 1)

    # Plot the dataset
    p1.scatter(X, y, marker='x', c='r')
    p1.set_title('Profit distribution')
    p1.set_xlabel('Population of City in 10,000s')
    p1.set_ylabel('Profit in $10,000s')

    # Manipulate the dataset, add ones to its left
    X = hstack((ones(shape=(m, 1)), X))
    theta = zeros(shape=(n, 1))

    # Parameters for gradient descent
    iter = 1500
    alpha = 0.01

    theta, J_history = gradient_descent(X, y, theta, alpha, iter)

    pred = X.dot(theta).flatten()
    p1.plot(X[:, 1:], pred)

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
    # print J_vals
    p2.contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
    p2.set_title('Contour, showing minimum')
    p2.set_xlabel('theta_0')
    p2.set_ylabel('theta_1')
    p2.scatter(theta[0][0], theta[1][0], marker='x', color='r')
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