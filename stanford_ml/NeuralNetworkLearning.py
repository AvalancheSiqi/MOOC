import numpy as np
import scipy.io as sio
from scipy import optimize
import matplotlib.pyplot as plt
from math import sqrt, floor, ceil

# Machine Learning by Andrew Ng on Coursera week 4
# Implement the backpropagation algorithm for neural networks and apply it to the task of hand-written digit recognition


def main():
    # Set up parameters for future usage
    input_layer_size = 400
    hidden_layer_size = 25
    num_labels = 10

    # == == == == == = Part 1: Loading and Visualizing Data == == == == == == = #
    print "Loading and Visualizing Data ...\n"
    dataset = sio.loadmat('datasets/ex4data1.mat')
    X = dataset['X']
    y = dataset['y']
    m, n = X.shape

    # Transfer y to matrix y_trans where the i-th column corresponds to number i
    y_trans = np.zeros(shape=(m, num_labels))
    for i in range(m):
        y_trans[i, y[i] - 1] = 1

    # Random select 100 data points to display
    rand_idx = np.random.permutation(m)
    sel = X[rand_idx[:100]][:]
    display_data(sel)

    # == == == == == == == == Part 2: Initializing Parameters == == == == == == = #
    print "Initializing Neural Network Parameters ...\n"
    initial_theta1 = rand_initialize_weights(input_layer_size, hidden_layer_size)
    initial_theta2 = rand_initialize_weights(hidden_layer_size, num_labels)
    # Unroll parameters
    initial_nn_params = np.hstack((initial_theta1.flatten(), initial_theta2.flatten()))

    # == == == == == == == == Part 3: Training NN == == == == == == = #
    print "Training Neural Network...\n"
    lam = 1
    iteration = 50
    nn_params, cost_min, _ = optimize.fmin_l_bfgs_b(nn_cost, initial_nn_params, fprime=nn_grad,
                                                    args=(input_layer_size, hidden_layer_size, num_labels, X, y_trans, lam),
                                                    maxiter=iteration)

    theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, (hidden_layer_size + 1))
    pred = predict(theta1, theta2, X)
    print "\nTraining Set Accuracy: %.4f\n" % (np.mean(pred == y.ravel()) * 100)


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


def rand_initialize_weights(l_in, l_out):
    """
    Initialize random weights for NN
    :param l_in: input layer size
    :param l_out: output layer size
    :return: random matrix with all values bounded within epsilon
    """
    epsilon = 0.12
    return np.random.random((l_out, l_in+1))*2*epsilon - epsilon


def sigmoid(z):
    """
    Map input to range zero to one
    :param z: predict value
    :return: mapped value between 0 and 1
    """
    return 1.0 / (1 + np.exp(-z))


def sigmoid_gradient(z):
    """
    Returns the gradient of the sigmoid function
    """
    return sigmoid(z)*(1-sigmoid(z))


def unroll_params(nn_params, input_layer_size, hidden_layer_size, num_labels):
    """
    Unroll parameters array to 2D matrix
    """
    # Reshape nn_params back to parameters theta1 and theta2
    theta1 = nn_params[:hidden_layer_size * (input_layer_size + 1)].reshape(hidden_layer_size, (input_layer_size + 1))
    theta2 = nn_params[hidden_layer_size * (input_layer_size + 1):].reshape(num_labels, (hidden_layer_size + 1))
    return theta1, theta2


def nn_cost(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y_trans, lam):
    """
    Cost function for neural network learning
    """
    theta1, theta2 = unroll_params(nn_params, input_layer_size, hidden_layer_size, num_labels)

    m, n = X.shape
    hidden = sigmoid(np.hstack((np.ones(shape=(m, 1)), X)).dot(theta1.T))
    output = sigmoid(np.hstack((np.ones(shape=(m, 1)), hidden)).dot(theta2.T))

    cost = -1.0/m*(y_trans*np.log(output) + (1-y_trans)*np.log(1-output)).sum() \
           + lam/2.0/m*((theta1[:, 1:]*theta1[:, 1:]).sum() + (theta2[:, 1:]*theta2[:, 1:]).sum())
    return cost


def nn_grad(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y_trans, lam):
    """
    Backpropagation implementation, gradient function for NN weights
    """
    theta1, theta2 = unroll_params(nn_params, input_layer_size, hidden_layer_size, num_labels)

    m, n = X.shape
    delta1, delta2 = 0, 0

    for i in range(m):
        a1 = X[i, :].reshape(1, n)    # 1 x 400
        z2 = np.hstack((np.ones(shape=(1, 1)), a1)).dot(theta1.T)  # 1 x 25
        a2 = sigmoid(z2)    # 1 x 25
        z3 = np.hstack((np.ones(shape=(1, 1)), a2)).dot(theta2.T)  # 1 x 10
        a3 = sigmoid(z3)    # 1 x 10

        err3 = a3 - y_trans[i, :] # 1 x 10
        err2 = err3.dot(theta2) * np.hstack((np.ones(shape=(1, 1)), sigmoid_gradient(z2)))  # 1 x 26
        err2 = err2[:, 1:]  # 1 x 25
        delta2 = delta2 + err3.T*np.hstack((np.ones(shape=(1, 1)), a2))    # 10 x 26
        delta1 = delta1 + err2.T*np.hstack((np.ones(shape=(1, 1)), a1))    # 25 x 401

    theta1_grad = delta1/m + lam/m*np.hstack((np.zeros(shape=(hidden_layer_size, 1)), theta1[:, 1:]))   # 25 x 401
    theta2_grad = delta2/m + lam/m*np.hstack((np.zeros(shape=(num_labels, 1)), theta2[:, 1:]))  # 10 x 26
    grad = np.concatenate((theta1_grad.flatten(), theta2_grad.flatten()))

    return grad


def predict(theta1, theta2, X):
    """
    Predict with 3 layers NN and input matrix X
    """
    m, n = X.shape
    a2 = sigmoid(np.hstack((np.ones(shape=(m, 1)), X)).dot(theta1.T))
    a3 = sigmoid(np.hstack((np.ones(shape=(m, 1)), a2)).dot(theta2.T))
    return np.argmax(a3, axis=1) + 1


if __name__ == '__main__':
    main()