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

X = np.array([[0],[0],[0],[1],[0]])
m=20

for t in range(m):
    X=P@X

print(X)

eigenvals,eigenvecs=np.linalg.eig(P)
print("eigenvals:",eigenvals)
print("eigenvecs",eigenvecs)

X_inf = eigenvecs[:,0]

print(f"Eigenvector corresponding to the eigenvalue 1:\n{X_inf[:,np.newaxis]}")

# This is organised as a function only for grading purposes.
def check_eigenvector(P, X_inf):
    ### START CODE HERE ###
    X_check = X1=P@X_inf
    ### END CODE HERE ###
    return X_check

X_check = check_eigenvector(P, X_inf)
print("Original eigenvector corresponding to the eigenvalue 1:\n" + str(X_inf))
print("Result of multiplication:" + str(X_check))

# Function np.isclose compares two NumPy arrays element by element, allowing for error tolerance (rtol parameter).
print("Check that PX=X element by element:" + str(np.isclose(X_inf, X_check, rtol=1e-10)))


X_inf = X_inf/sum(X_inf)
print(f"Long-run probabilities of being at each webpage:\n{X_inf[:,np.newaxis]}")


imgs=utils.load_images('./data/')
height,width=imgs[0].shape
print(f'\n Your dataset has {len(imgs)} images of size {height}x{width} pixels\n')
plt.imshow(imgs[0],cmap='gray')

imgs_flatten=np.array([im.reshape(-1) for im in imgs])
print(f'imgs_flatten shape: {imgs_flatten.shape}')