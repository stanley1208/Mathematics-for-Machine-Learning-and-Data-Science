import numpy as np


A=np.array([[4,9,9],[9,1,6],[9,2,3]])
print("Matrix A (3 by 3):\n",A)
B=np.array([[2,2],[5,7],[4,4]])
print("Matrix B (2 by 2):\n",B)

print(np.matmul(A,B))
print(A @ B)

try:
    np.matmul(B,A)
except ValueError as err:
    print(err)

try:
    B @ A
except ValueError as err:
    print(err)

x=np.array([1,-2,-5])
y=np.array([4,3,-1])

print("Shape of x matrix is:",x.shape)
print("Number of dimension of x:",x.ndim)
print("reshape x:",x.reshape((3,1)).shape)
print("Number of reshape dimension of x:",x.reshape((3,1)).ndim)

print(np.matmul(x,y))

try:
    np.matmul(x.reshape((3,1)),y.reshape((3,1)))
except ValueError as err:
    print(err)

print(np.dot(A,B))

print(A-2)
