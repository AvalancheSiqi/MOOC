import scipy.io as sio
import numpy as np
from math import sqrt, floor, ceil
import matplotlib.pyplot as plt
from scipy import optimize

# Machine Learning by Andrew Ng on Coursera week 3
# One-vs-all logistic regression to recognize hand-written digits.


def main():
    # == == == == == = Part 1: Loading and Visualizing Data == == == == == == = #
    print "Loading and Visualizing Data ..."
    dataset = sio.loadmat('datasets/ex3data1.mat')
    X = dataset['X']
    y = dataset['y']
    num_labels = 10
    m, n = X.shape

    # Random select 100 data points to display
    rand_idx = np.random.permutation(m)
    sel = X[rand_idx[:100]][:]
    display_data(sel)

    # == == == == == = Part 2: Vectorize Logistic Regression == == == == == == = #
    print "Training One-vs-All Logistic Regression..."

    lam = 0.01
    X = np.hstack((np.ones(shape=(m, 1)), X))
    all_theta = one_vs_all(X, y, num_labels, lam)

    # == == == == == = Part 3: Predict for One-Vs-All == == == == == == = #
    pred = predict_one_vs_all(all_theta, X)
    print "\nTraining Set Accuracy: %.4f\n" % (np.mean(pred==y.ravel())*100)


def display_data(X, example_width=None):
    """
    Displays 2D data stored in X in a nice grid
    :param X: Hand writing numbers array presentation
    :param example_width: plot example width
    :return: figure with 2D hand writing numbers data plotted
    """
    if not example_width:
        example_width = round(sqrt(X.shape[1]))

    m, n = X.shape
    example_height = n / example_width

    # Compute number of items to display
    display_rows = int(floor(sqrt(m)))
    display_cols = int(ceil(m/display_rows))

    # Between images padding
    pad = 1

    # Set up blank figure to display, with size satisfies
    display_array = -1*np.ones((pad+display_rows*(example_height+pad), pad+display_cols*(example_width+pad)))

    # Copy each example into a patch on the display array
    curr_ex = 0
    for i in range(display_rows):
        for j in range(display_cols):
            if curr_ex >= m:
                break
            max_val = max(abs(X[curr_ex]))
            top = pad+j*(example_height+pad)
            left = pad+i*(example_width+pad)
            display_array[top:top+example_height, left:left+example_width] \
                = X[curr_ex].reshape(example_height, example_width)/max_val
            curr_ex += 1

    # Plot the image
    plt.imshow(display_array.T, cmap="Greys_r")
    plt.show()
    return


def one_vs_all(X, y, num_labels, lam):
    """
    Trains multiple logistic regression classifiers and returns all the classifiers in a matrix all_theta,
    where the i-th row of all_theta corresponds to the classifier for label i
    :param X: training input matrix
    :param y: target result
    :param num_labels: number of multiple classes
    :param lam: coefficient of regularisation
    :return: classifiers that represent multiple classifications
    """
    m, n = X.shape
    all_theta = np.ones(shape=(num_labels, n))

    for i in range(num_labels):
        all_theta[i, :] = optimize.fmin_cg(compute_cost_reg, all_theta[i, :], fprime=grad,
                                            args=(X, y==(i+1), lam), maxiter=50)
    return all_theta


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
    :param lam: coefficient of regularisation
    :return: ost function value
    """
    m = y.size
    pred = sigmoid(np.dot(X, theta))
    return -1.0/m*(np.dot(y.T, np.log(pred)) + np.dot((1-y).T, np.log(1-pred))) + lam/2.0/m*sum(theta[1:]*theta[1:])


def grad(theta, X, y, lam):
    """
    Gradient for a single example
    """
    m = y.size
    theta = theta.reshape(len(theta), 1)
    pred = sigmoid(np.dot(X, theta)).reshape(m, 1)
    error = np.dot(X.T, (pred - y))
    ret = (error/m + 1.0*lam/m*theta)
    ret[0] -= lam/m*theta[0]
    return ret.flatten()


def predict_one_vs_all(all_theta, X):
    """
    Predict the label for a trained one-vs-all classifier
    :param all_theta: one-vs-all classifier coefficients
    :param X: training input matrix
    :return: predict value which has the highest probability
    """
    return np.argmax(np.dot(X, all_theta.T), axis=1)+1


if __name__ == '__main__':
    main()