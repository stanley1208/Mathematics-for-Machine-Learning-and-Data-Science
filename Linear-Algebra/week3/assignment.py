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