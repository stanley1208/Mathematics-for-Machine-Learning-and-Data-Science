import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import utils
import w4_unittest




P = np.array([

    [0, 0.75, 0.35, 0.25, 0.85],
    [0.15, 0, 0.35, 0.25, 0.05],
    [0.15, 0.15, 0, 0.25, 0.05],
    [0.15, 0.05, 0.05, 0, 0.05],
    [0.55, 0.05, 0.25, 0.25, 0]
])

X0 = np.array([[0], [0], [0], [1], [0]])

### START CODE HERE ###

# Multiply matrix P and X_0 (matrix multiplication).
X1 = np.dot(P, X0)

### END CODE HERE ###

print(f'Sum of columns of P: {sum(P)}')
print(f'X1:\n{X1}')