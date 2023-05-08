import numpy as np



### START CODE HERE ###
A = np.array([
        [2, -1, 1, 1],
        [1, 2, -1, -1],
        [-1, 2, 2, 2],
        [1, -1, 2, 1]
    ], dtype=np.dtype(float))
b = np.array([6, 3, 14, 8], dtype=np.dtype(float))

### END CODE HERE ###


### START CODE HERE ###
# determinant of matrix A
d = np.linalg.det(A)

# solution of the system of linear equations
# with the corresponding coefficients matrix A and free coefficients b
x = np.linalg.solve(A,b)
### END CODE HERE ###

print(f"Determinant of matrix A: {d:.2f}")

print(f"Solution vector: {x}")