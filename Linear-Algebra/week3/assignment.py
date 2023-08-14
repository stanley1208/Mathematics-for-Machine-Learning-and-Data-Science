import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
import pandas as pd
import w3_tools

np.random.seed(3)

m=30

X,Y=make_regression(n_samples=m,n_features=1,noise=20,random_state=1)

X=X.reshape((1,m))
Y=Y.reshape((1,m))

print("Training dataset X:")
print(X)
print("Training dataset Y:")
print(Y)

plt.scatter(X,Y,c='black')
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.show()


### START CODE HERE ### (~ 3 lines of code)
# Shape of variable X.
shape_X = np.shape(X)
# Shape of variable Y.
shape_Y = np.shape(Y)
# Training set size.
m = 30
### END CODE HERE ###

print ('The shape of X: ' + str(shape_X))
print ('The shape of Y: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))


# GRADED FUNCTION: layer_sizes

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """
    ### START CODE HERE ### (~ 2 lines of code)
    # Size of input layer.
    n_x = X.shape[0]
    # Size of output layer.
    n_y = Y.shape[0]
    ### END CODE HERE ###
    return (n_x, n_y)


(n_x, n_y) = layer_sizes(X, Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the output layer is: n_y = " + str(n_y))


# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_y):
    """
    Returns:
    params -- python dictionary containing your parameters:
                    W -- weight matrix of shape (n_y, n_x)
                    b -- bias value set as a vector of shape (n_y, 1)
    """

    ### START CODE HERE ### (~ 2 lines of code)
    W = np.random.randn(n_y,n_x)*0.01
    b = np.zeros((n_y,1))
    ### END CODE HERE ###

    assert (W.shape == (n_y, n_x))
    assert (b.shape == (n_y, 1))

    parameters = {"W": W,
                  "b": b}

    return parameters


parameters = initialize_parameters(n_x, n_y)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))