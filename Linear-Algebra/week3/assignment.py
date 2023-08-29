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


# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    Y_hat -- The output
    """
    # Retrieve each parameter from the dictionary "parameters".
    ### START CODE HERE ### (~ 2 lines of code)
    W = parameters["W"]
    b = parameters["b"]
    ### END CODE HERE ###

    # Implement Forward Propagation to calculate Z.
    ### START CODE HERE ### (~ 2 lines of code)
    Z = np.dot(W,X)+b
    Y_hat = Z
    ### END CODE HERE ###

    assert (Y_hat.shape == (n_y, X.shape[1]))

    return Y_hat


Y_hat = forward_propagation(X, parameters)

print(Y_hat)


def compute_cost(Y_hat, Y):
    """
    Computes the cost function as a sum of squares

    Arguments:
    Y_hat -- The output of the neural network of shape (n_y, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)

    Returns:
    cost -- sum of squares scaled by 1/(2*number of examples)

    """
    # Number of examples.
    m = Y.shape[1]

    # Compute the cost function.
    cost = np.sum((Y_hat - Y) ** 2) / (2 * m)

    return cost


print("cost = " + str(compute_cost(Y_hat, Y)))

parameters=w3_tools.train_nn(parameters,Y_hat,X,Y)

print("W="+str(parameters["W"]))
print("b="+str(parameters["b"]))


# GRADED FUNCTION: nn_model

def nn_model(X, Y, num_iterations=10, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    num_iterations -- number of iterations in the loop
    print_cost -- if True, print the cost every iteration

    Returns:
    parameters -- parameters learnt by the model. They can then be used to make predictions.
    """

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[1]

    # Initialize parameters
    ### START CODE HERE ### (~ 1 line of code)
    parameters = initialize_parameters(n_x,n_y)
    ### END CODE HERE ###

    # Loop
    for i in range(0, num_iterations):

        ### START CODE HERE ### (~ 2 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "Y_hat".
        Y_hat = forward_propagation(X,parameters)

        # Cost function. Inputs: "Y_hat, Y". Outputs: "cost".
        cost = compute_cost(Y_hat,Y)
        ### END CODE HERE ###

        # Parameters update.
        parameters = w3_tools.train_nn(parameters, Y_hat, X, Y)

        # Print the cost every iteration.
        if print_cost:
            print("Cost after iteration %i: %f" % (i, cost))

    return parameters

parameters = nn_model(X, Y, num_iterations=15, print_cost=True)
print("W = " + str(parameters["W"]))
print("b = " + str(parameters["b"]))

W_simple = parameters["W"]
b_simple = parameters["b"]


X_pred=np.array([-0.95,0.2,1.5])

fig,ax=plt.subplots()
plt.scatter(X,Y,color="black")

plt.xlabel("$x$")
plt.ylabel("$y$")

X_line=np.arange(np.min(X[0,:]),np.max(X[0,:])*1.1,0.1)
ax.plot(X_line,W_simple[0,0]*X_line+b_simple[0,0],"r")
ax.plot(X_pred,W_simple[0,0]*X_pred+b_simple[0,0],"bo")
plt.plot()
plt.show()