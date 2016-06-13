#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import sys
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
import scipy.io as sio
from math import ceil
import random

# Machine Learning by Andrew Ng on Coursera week 6
# Spam Classification with SVMs


def main():
    # ==================== Part 1: Email Preprocessing ====================
    print "Preprocessing sample email (emailSample1.txt)\n"
    # Extract features, use read() as we know it won't yield EOF
    file_contents = open('datasets/emailSample1.txt').read()
    # Load Vocabulary
    vl = get_vocab_list()
    word_indices = process_email(file_contents, vl)
    print "Word Indices\n", word_indices, "\n"

    # ==================== Part 2: Features Extraction ====================
    print "Extracting features from sample email (emailSample1.txt)\n"
    features = email_features(word_indices, len(vl))
    print "Length of feature vector: %d\n" % len(features)
    print "Number of non-zero entries: %d\n" % np.sum(features)

    # ========= Part 3: Train Linear SVM for Spam Classification ==========
    dataset = sio.loadmat('datasets/spamTrain.mat')
    X = dataset['X'].astype(float)
    y = dataset['y'].astype('int8')
    print 'Training Linear SVM (Spam Classification)\n'
    print '(this may take 1 to 2 minutes) ...\n'
    C = 0.1
    model = train_svm(X, y, C, linear_kernel)
    p = predict_svm(X, model)
    print '\nTraining Accuracy: %.4f\n' % (np.mean(p == y)*100)

    # ================= Part 4: Test Spam Classification ==================
    dataset_test = sio.loadmat('datasets/spamTest.mat')
    Xtest = dataset_test['Xtest'].astype(float)
    ytest = dataset_test['ytest'].astype('int8')
    print 'Evaluating the trained Linear SVM on a test set ...\n'
    p_test = predict_svm(Xtest, model)
    print 'Test Accuracy: %.4f\n' % (np.mean(p_test == ytest) * 100)

    # ================== Part 5: Top Predictors of Spam ===================
    print 'Top predictors of spam: \n'
    weights = model.w.ravel()
    idx = np.argsort(weights)[::-1]
    vl2 = np.loadtxt('datasets/vocab.txt', delimiter='\t', dtype=str, usecols=[1])
    for i in range(15):
        print '%-15s (%f)' % (vl2[idx[i]], weights[idx[i]])

    # ================== Part 6: Try Your Own Emails ===================
    filename1 = 'datasets/emailSample1.txt'
    test_email(filename1, model, vl)
    filename2 = 'datasets/emailSample2.txt'
    test_email(filename2, model, vl)
    filename3 = 'datasets/spamSample1.txt'
    test_email(filename3, model, vl)
    filename4 = 'datasets/spamSample2.txt'
    test_email(filename4, model, vl)
    print '(1 indicates spam, 0 indicates not spam)\n\n'


def get_vocab_list():
    """
    Get the vocabulary list as a dict with word as key and idx as value
    """
    vl = np.loadtxt('datasets/vocab.txt', delimiter='\t', dtype=str)
    return {k: int(v) for (v, k) in vl}


def process_email(email_contents, vl):
    """
    Process email contents then tokenize, stem and map them into vocabulary list
    """
    # Lower case
    email_contents = email_contents.lower()
    # Strip all HTML
    email_contents = re.sub(r'<[^<>]+>', ' ', email_contents)
    # Handle Numbers
    email_contents = re.sub(r'\d+', 'number', email_contents)
    # Handle URLs
    email_contents = re.sub(r'(http|https)://[^\s]*', 'httpaddr', email_contents)
    # Handle Email Addresses
    email_contents = re.sub(r'[^\s]+@[^\s]+', 'emailaddr', email_contents)
    # Handle $ sign
    email_contents = re.sub(r'[$]+', 'dollar', email_contents)

    # == == == == == == == == == == == == == Tokenize Email == == == == == == == == == == == == == =
    print "\n===== Processed Email =====\n"
    token = RegexpTokenizer(r'\w+').tokenize(email_contents)
    stemer = PorterStemmer()
    token = map(stemer.stem, token)
    print map(str, token)
    print "\n=========================\n"

    word_indices = [vl.get(str(k), None) for k in token]
    word_indices = [x for x in word_indices if x is not None]

    return word_indices


def email_features(word_indices, dict_len):
    """
    Create features matrix with each exist word in dict marked as 1 otherwise 0
    """
    ret = np.zeros(shape=(1, dict_len))
    for i in word_indices:
        ret[0, i-1] = 1
    return ret


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


def test_email(filename, model, vl):
    """
    Test whether a given file is spam or not based on given model and vocabulary list
    """
    file_contents = open(filename).read()
    word_indices = process_email(file_contents, vl)
    features = email_features(word_indices, len(vl))
    p = predict_svm(features, model)
    print 'Processed %s\nSpam Classification: %d\n' % (filename, p)


if __name__ == '__main__':
    main()
