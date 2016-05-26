import scipy.io as sio
import numpy as np
from math import sqrt, floor, ceil
import matplotlib.pyplot as plt

# Machine Learning by Andrew Ng on Coursera week 3
# One-vs-all logistic regression to recognize hand-written digits.


def main():
    # == == == == == = Part 1: Loading and Visualizing Data == == == == == == = #
    print "Loading and Visualizing Data ..."
    dataset = sio.loadmat('datasets/ex3data1.mat')
    X = dataset['X']
    y = dataset['y']
    m, n = X.shape

    # Random select 100 data points to display
    rand_idx = np.random.permutation(m)
    sel = X[rand_idx[:100]][:]
    display_data(sel)


def display_data(X, example_width=None):
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

    # sio.savemat('datasets/temp.mat', {'X':display_array})
    plt.figure()


if __name__ == '__main__':
    main()