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

    fig = plt.figure(figsize=(12, 16))
    # Unfitted linear regression
    p1 = fig.add_subplot(321)
    p1.set_title('Unfitted linear regression')
    # Linear regression learning curve
    p2 = fig.add_subplot(322)
    p2.set_title('Linear regression learning curve')
    # Polynomial regression
    p3 = fig.add_subplot(323)
    p3.set_title('Polynomial regression')
    # Polynomial regression learning curve
    p4 = fig.add_subplot(324)
    p4.set_title('Polynomial regression learning curve')
    # Display train, cv and test datasets
    p5 = fig.add_subplot(325)
    p5.set_title('Polynomial regression with best lambda')
    # Selecting best lambda
    p6 = fig.add_subplot(326)
    p6.set_title('Selecting lambda using a cross validation set')

    for p in (p1, p3):
        p.scatter(X, y, marker='x', color='r', linewidths=1.5)
        p.set_xlabel('Change in water level (x)')
        p.set_ylabel('Water flowing out of the dam (y)')

    for p in (p2, p4):
        p.set_xlabel('Number of training example')
        p.set_ylabel('Error')
        p.set_xlim([0, 13])

    p5.scatter(X, y, marker='x', color='r', linewidths=1.5)
    p5.scatter(Xval, yval, marker='+', color='b', linewidths=1.5)
    p5.scatter(Xtest, ytest, marker='o', color='g', linewidths=1.5)

    # == == == == == == == == Part 2: Train Linear Regression == == == == == == = #
    X = np.hstack((np.ones(shape=(m, 1)), X))
    Xval = np.hstack((np.ones(shape=(Xval.shape[0], 1)), Xval))
    Xtest = np.hstack((np.ones(shape=(Xtest.shape[0], 1)), Xtest))
    lam = 1
    theta_lin, _ = train_linear_reg(X, y, lam)
    theta_lin = theta_lin.reshape(theta_lin.size, 1)

    # Plot fit over the data
    line_points = np.arange(-50, 50, 10)
    p1.plot(line_points, np.hstack((np.ones(shape=(10, 1)), line_points.reshape(10, 1))).dot(theta_lin), marker='.')

    # == == == == == == Part 3: Plot Linear Regression Learning Curve == == == == = #
    plot_learning_curve(p2, X, y, Xval, yval, lam)

    # == == == == == == == == Part 4: Train Polynomial Regression == == == == == == = #
    lam = 1
    X_poly = normalise_feature(poly_feature(X, 8))[0]
    theta_poly, _ = train_linear_reg(X_poly, y, lam)
    pred_poly = X_poly.dot(theta_poly)
    sorted_X, sorted_pred_poly = (list(t) for t in zip(*sorted(zip(X[:, 1:], pred_poly))))
    # Plot fit over the data
    p3.plot(sorted_X, sorted_pred_poly, marker='.')

    # == == == == == == Part 5: Plot Polynomial Regression Learning Curve == == == == = #
    Xval_poly = normalise_feature(poly_feature(Xval, 8))[0]
    plot_learning_curve(p4, X_poly, y, Xval_poly, yval, lam)

    # == == == == == == Part 6: Select the best lambda value == == == == = #
    lam_list = [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
    best_lambda, best_err = select_best_lambda(p6, X_poly, y, Xval_poly, yval, lam_list)
    Xtest_poly = normalise_feature(poly_feature(Xtest, 8))[0]
    best_theta, _ = train_linear_reg(Xtest_poly, ytest, best_lambda)
    pred_test_poly = Xtest_poly.dot(best_theta)
    sorted_Xtest, sorted_pred_test_poly = (list(t) for t in zip(*sorted(zip(Xtest[:, 1:], pred_test_poly))))
    # Plot fit over the data
    p5.plot(sorted_Xtest, sorted_pred_test_poly, marker='.', color='g')
    best_test = cost(best_theta, Xtest_poly, ytest, lam)
    print "Best regularisation value should be %.4f with validation error %.4f, test error %.4f"\
          % (best_lambda, best_err, best_test)

    plt.tight_layout()
    plt.show()
    return


def poly_feature(X, deg):
    """
    Map features to polynomial features on degree deg
    """
    m = X.shape[0]
    ret = np.ones(shape=(m, deg+1))
    for i in range(deg):
        ret[:, i+1] = ret[:, i]*X[:, 1]
    return ret


def normalise_feature(X):
    """
    Normalise the features of input matrix
    """
    (m, n) = X.shape
    ms = np.ones(shape=(n, 2))
    for j in range(1, n):
        ms[j, 0] = np.mean(X[:, j])
        ms[j, 1] = np.std(X[:, j])
        X[:, j] = (X[:, j] - ms[j, 0]) / ms[j, 1]
    return X, ms


def cost(theta, X, y, lam):
    """
    Cost function of linear regression with regularisation
    """
    m, n = X.shape
    theta = theta.reshape(n, 1)
    return 1.0/2/m*sum((X.dot(theta) - y)**2) + lam/2.0/m*sum(theta[1:]**2)


def grad(theta, X, y, lam):
    """
    Gradient descent of linear regression with regularisation
    """
    m, n = X.shape
    theta = theta.reshape(n, 1)
    err = np.dot(X.T, X.dot(theta) - y)
    ret = err/m + 1.0*lam/m*theta
    ret[0] -= 1.0*lam/m*theta[0]
    return ret.flatten()


def train_linear_reg(X, y, lam):
    """
    Train the linear regression model
    """
    m, n = X.shape
    initial_theta = np.zeros(shape=(n, 1))
    theta, error, _, _, _ = optimize.fmin_cg(cost, initial_theta, fprime=grad, args=(X, y, lam), maxiter=200,
                                             full_output=True, disp=False)
    return theta, error


def plot_learning_curve(p, X, y, Xval, yval, lam):
    """
    Plot learning curve to show the difference change of training data and cv data
    """
    m, n = X.shape
    theta = np.zeros(shape=(m-1, n))
    error_train = np.empty(m-1)
    error_cv = np.empty(m-1)
    for i in range(m-1):
        theta[i, :], error_train[i] = train_linear_reg(X[:i+2, :], y[:i+2], lam)
        error_cv[i] = cost(theta[i, :], Xval, yval, lam)
    p.plot(range(2, m+1), error_train)
    p.plot(range(2, m+1), error_cv)
    return


def select_best_lambda(p, X, y, Xval, yval, lam_list):
    """
    Choose the best regularisation value based on cv error
    """
    error_train = np.empty(len(lam_list))
    error_cv = np.empty(len(lam_list))
    best_lambda = 0
    best_err = float("inf")
    for i, lam in enumerate(lam_list):
        theta, error_train[i] = train_linear_reg(X, y, lam)
        error_cv[i] = cost(theta, Xval, yval, lam)
        if error_cv[i] < best_err:
            best_err = error_cv[i]
            best_lambda = lam
    p.plot(lam_list, error_train)
    p.plot(lam_list, error_cv)
    return best_lambda, best_err


if __name__ == '__main__':
    main()
