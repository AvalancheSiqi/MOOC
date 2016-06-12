import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from math import ceil
import random
import sys

# Machine Learning by Andrew Ng on Coursera week 6
# Support vector machines (SVMs) with various example 2D datasets


def main():
    fig = plt.figure(figsize=(16, 5))

    # == == == == == = Part 1: Loading and Visualizing Data == == == == == == = #
    print "Loading and Visualizing Data 1 ...\n"
    X1, y1 = load_data('datasets/ex6data1.mat')
    p1 = fig.add_subplot(131)
    plot_data(p1, X1, y1)

    # == == == == == == = Part 2: Training Linear SVM == == == == == == == == = #
    print "Training Linear SVM ...\n"
    C = 1
    model1 = train_svm(X1, y1, C, linear_kernel, tol=1e-3, max_iter=20)
    visualize_boundary_linear(p1, X1, model1)

    # == == == == == = Part 3: Training SVM with RBF Kernel == == == == == == = #
    print "Loading and Visualizing Data 2 ...\n"
    X2, y2 = load_data('datasets/ex6data2.mat')
    p2 = fig.add_subplot(132)
    plot_data(p2, X2, y2)

    print "Training Gaussian SVM model2 ...\n"
    C = 1
    sigma = 0.1
    model2 = train_svm(X2, y2, C, gaussian_kernel, sigma=sigma)
    visualize_boundary(p2, X2, model2)

    # == == == == == = Part 4: Training SVM with RBF Kernel == == == == == == = #
    print "Loading and Visualizing Data 3 ...\n"
    dataset = sio.loadmat('datasets/ex6data3.mat')
    X3 = dataset['X'].astype(float)
    y3 = dataset['y'].astype('int8')
    Xval3 = dataset['Xval'].astype(float)
    yval3 = dataset['yval'].astype('int8')
    p3 = fig.add_subplot(133)
    plot_data(p3, X3, y3)

    print "Training Gaussian SVM model3 ...\n"
    C, sigma = select_best_param(X3, y3, Xval3, yval3)
    model3 = train_svm(X3, y3, C, gaussian_kernel, sigma=sigma)
    print "Best parameters for dataset3 are C: %f, sigma: %f\n" % (C, sigma)
    visualize_boundary(p3, X3, model3)

    plt.tight_layout()
    plt.show()
    return


def load_data(path):
    """
    Load dataset from mat file, convert y to int8 type
    """
    dataset = sio.loadmat(path)
    X = dataset['X'].astype(float)
    y = dataset['y'].astype('int8')
    return X, y


def plot_data(p, X, y):
    """
    Plot scatter data X with label y on picture p
    """
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p.scatter(X[pos[0]].T[0], X[pos[0]].T[1], marker='o', c='g')
    p.scatter(X[neg[0]].T[0], X[neg[0]].T[1], marker='x', c='r')
    return


class Model:
    """
    Model class, store weights, bias and kernel function, etc.
    """
    def __init__(self, X, y, w, b, alphas, kernel, sigma):
        self.X = X
        self.y = y
        self.w = w
        self.b = b
        self.alphas = alphas
        self.kernel = kernel
        self.sigma = sigma


def linear_kernel(x1, x2):
    m = x1.size
    return np.dot(x1.reshape(1, m), x2.reshape(m, 1))


def gaussian_kernel(x1, x2, sigma):
    return np.exp(-(x1-x2)**2/2.0/(sigma**2))


def train_svm(X, y, C, kernel, sigma=None, tol=None, max_iter=None):
    """
    Trains an SVM classifier using a simplified version of the SMO algorithm.
    :param X: Matrix of training examples
    :param y: Column matrix containing 1 for positive examples and 0 for negative examples
    :param C: Standard SVM regularization parameter
    :param kernel: Kernel function
    :param sigma: Sigma variable used in Gaussian Kernel
    :param tol: Tolerance value used for determining equality of floating point numbers
    :param max_iter: Number of iterations over the dataset
    :return: A trained SVM classifier stored in Model model
    """
    if not tol:
        tol = 1e-3
    if not max_iter:
        max_iter = 5

    # variables
    m, n = X.shape
    y2 = np.ones(shape=(m, 1))
    y2[np.where(y==0)[0]] = -1
    alphas = np.zeros(shape=(m, 1))
    b = 0
    E = np.zeros(shape=(m, 1))
    eta = 0
    L = 0
    H = 0
    passes = 0

    if kernel.__name__ == 'linear_kernel':
        K = X.dot(X.T)
    if kernel.__name__ == 'gaussian_kernel':
        if not sigma:
            sigma = 0.1
        X2 = np.sum(X**2, axis=1).reshape(m, 1)
        K = X2 + X2.T - 2*X.dot(X.T)
        K = gaussian_kernel(1, 0, sigma)**K

    sys.stdout.write("train SVM...")
    sys.stdout.flush()
    dots = 12
    while passes < max_iter:
        num_changed_alphas = 0
        for i in range(m):
            E[i] = b + (alphas*y2*(K[:, i].reshape(m, 1))).sum() - y2[i]
            if (y2[i]*E[i] < -tol and alphas[i, 0] < C) or (y2[i]*E[i] > tol and alphas[i, 0] > 0):
                j = ceil(m * random.random()) - 1
                while j == i:
                    j = ceil(m * random.random()) - 1

                E[j] = b + (alphas*y2*K[:, j].reshape(m, 1)).sum() - y2[j]

                alpha_i_old = alphas[i, 0]
                alpha_j_old = alphas[j, 0]

                if y2[i] == y2[j]:
                    L = max(0, alphas[j, 0]+alphas[i, 0]-C)
                    H = min(C, alphas[j, 0]+alphas[i, 0])
                else:
                    L = max(0, alphas[j, 0]-alphas[i, 0])
                    H = min(C, C+alphas[j, 0]-alphas[i, 0])

                if L == H:
                    continue

                eta = 2*K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:
                    continue

                alphas[j, 0] -= (y2[j]*(E[i]-E[j]))/eta
                alphas[j, 0] = min(H, alphas[j, 0])
                alphas[j, 0] = max(L, alphas[j, 0])

                if abs(alphas[j, 0] - alpha_j_old) < tol:
                    alphas[j, 0] = alpha_j_old
                    continue

                alphas[i, 0] = alphas[i, 0] + y2[i]*y2[j]*(alpha_j_old-alphas[j, 0])
                b1 = b - E[i] - y2[i]*(alphas[i, 0]-alpha_i_old)*K[i, j].T - y2[j]*(alphas[j, 0]-alpha_j_old)*K[i, j].T
                b2 = b - E[j] - y2[i]*(alphas[i, 0]-alpha_i_old)*K[i, j].T - y2[j]*(alphas[j, 0]-alpha_j_old)*K[j, j].T

                if 0 < alphas[i, 0] < C:
                    b = b1
                elif 0 < alphas[j, 0] < C:
                    b = b2
                else:
                    b = (b1+b2)/2

                num_changed_alphas += 1

        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0

        sys.stdout.write('.')
        sys.stdout.flush()
        dots += 1
        if dots > 78:
            dots = 0
            print

    print
    idx = (alphas > 0).ravel()
    w = (X.T).dot(alphas*y2)
    model = Model(X[idx], y2[idx], w, b, alphas[idx], kernel, sigma)
    return model


def predict_svm(X, model):
    """
    A vector of predictions using a trained SVM model (svmTrain)
    :param X: A mxn matrix where there each example is a row
    :param model: A svm model returned from svmTrain
    :return: A m x 1 column of predictions of {0, 1} values
    """
    m = X.shape[0]
    p = np.zeros(shape=(m, 1))
    if model.kernel.__name__ == 'linear_kernel':
        p = X.dot(model.w) + model.b
    if model.kernel.__name__ == 'gaussian_kernel':
        m2 = model.X.shape[0]
        X1 = np.sum(X**2, axis=1).reshape(m, 1)
        X2 = np.sum(model.X**2, axis=1).reshape(1, m2)
        K = X1 + (X2-2*X.dot(model.X.T))
        K = gaussian_kernel(1, 0, model.sigma)**K
        K = np.multiply(model.y.T, K)
        K = np.multiply(model.alphas.T, K)
        p = np.sum(K, axis=1)
    return (p >= 0).astype(int).reshape(m, 1)


def select_best_param(X, y, Xval, yval):
    """
    Select the optimal (C, sigma) learning parameters to use for SVM with RBF kernel
    """
    values = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30]
    opt_err = float("inf")
    for i in values:
        for j in values:
            model = train_svm(X, y, i, gaussian_kernel, sigma=j)
            predictions = predict_svm(Xval, model)
            pred_err = np.mean(predictions != yval)
            if opt_err > pred_err:
                opt_err = pred_err
                C = i
                sigma = j
    return C, sigma


def visualize_boundary_linear(p, X, model):
    """
    plots a linear decision boundary learned by the SVM
    """
    w = model.w
    b = model.b
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = -(w[0, 0]*xp+b) / w[1, 0]
    p.plot(xp, yp)
    return


def visualize_boundary(p, X, model):
    """
    Plot a non-linear decision boundary learned by the SVM
    """
    x1plot = np.linspace(min(X[:, 0]), max(X[:, 0]), 100).T
    x2plot = np.linspace(min(X[:, 1]), max(X[:, 1]), 100).T
    X1, X2 = np.meshgrid(x1plot, x2plot)
    m = X1.shape[0]
    vals = np.zeros(shape=X1.shape)
    for i in range(X1.shape[1]):
        this_X = np.hstack((X1[:, i].reshape(m, 1), X2[:, i].reshape(m, 1)))
        vals[:, i] = predict_svm(this_X, model).ravel()
    p.contour(X1, X2, vals, colors='b')
    return


if __name__ == '__main__':
    main()