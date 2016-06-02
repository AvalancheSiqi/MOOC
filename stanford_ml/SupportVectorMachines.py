import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from math import ceil
import random

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
    # model1 = train_svm(X1, y1, C, 'linear', 1e-3, 20)
    # visualize_boundary_linear(p1, X1, model1)

    # == == == == == = Part 3: Training SVM with RBF Kernel == == == == == == = #
    print "Loading and Visualizing Data 2 ...\n"
    X2, y2 = load_data('datasets/ex6data2.mat')
    p2 = fig.add_subplot(132)
    plot_data(p2, X2, y2)

    print "Training Gaussian SVM ...\n"
    C = 1
    model2 = train_svm(X2, y2, C, 'gaussian')
    np.savetxt('temp.txt', np.hstack((predict_svm(X2, model2), y2)))
    # visualize_boundary(p2, X2, y2, model2)

    # == == == == == = Part 4: Training SVM with RBF Kernel == == == == == == = #
    print "Loading and Visualizing Data 3 ...\n"
    X3, y3 = load_data('datasets/ex6data3.mat')
    p3 = fig.add_subplot(133)
    plot_data(p3, X3, y3)

    plt.tight_layout()
    plt.show()
    return


def load_data(path):
    dataset = sio.loadmat(path)
    X = dataset['X']
    y = dataset['y']
    y = y.astype('int8')
    return X, y


def plot_data(p, X, y):
    pos = np.where(y == 1)
    neg = np.where(y == 0)
    p.scatter(X[pos[0]].T[0], X[pos[0]].T[1], marker='o', c='g')
    p.scatter(X[neg[0]].T[0], X[neg[0]].T[1], marker='x', c='r')
    return


class Model:
    def __init__(self, X, y, w, b, alphas, kernel):
        self.X = X
        self.y = y
        self.w = w
        self.b = b
        self.alphas = alphas
        self.kernel = kernel


def gaussian_kernel(x1, x2, sigma=None):
    if not sigma:
        sigma = 0.1
    return np.exp(-(x1-x2)**2/2.0/(sigma**2))


def train_svm(X, y, C, kernel, tol=None, max_iter=None):
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

    if kernel == 'linear':
        K = X.dot(X.T)
    if kernel == 'gaussian':
        sigma = 0.1
        X2 = np.sum(X**2, axis=1).reshape(m, 1)
        K = X2 + X2.T - 2*X.dot(X.T)
        K = gaussian_kernel(1, 0, sigma)**K
    print "train SVM..."

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

    print "DONE training SVM...\n"

    idx = (alphas > 0).ravel()
    w = (X.T).dot(alphas*y2)
    model = Model(X[idx], y2[idx], w, b, alphas[idx], kernel)
    return model


def predict_svm(X, model):
    m = X.shape[0]
    p = np.zeros(shape=(m, 1))
    if model.kernel == 'linear':
        p = X.dot(model.w) + model.b
    if model.kernel == 'gaussian':
        m2 = model.X.shape[0]
        X1 = np.sum(X**2, axis=1).reshape(m, 1)
        X2 = np.sum(model.X**2, axis=1).reshape(1, m2)
        K = X1 + (X2-2*X.dot(model.X.T))
        K = gaussian_kernel(1, 0)**K
        K = np.multiply(model.y.T, K)
        K = np.multiply(model.alphas.T, K)
        p = np.sum(K, axis=1)
    return (p >= 0).astype(int).reshape(m, 1)


def visualize_boundary_linear(p, X, model):
    w = model.w
    b = model.b
    xp = np.linspace(min(X[:, 0]), max(X[:, 0]), 100)
    yp = -(w[0, 0]*xp+b) / w[1, 0]
    p.plot(xp, yp)
    return


def visualize_boundary(p, X, y, model):
    w = model.w
    b = model.b
    return


if __name__ == '__main__':
    main()