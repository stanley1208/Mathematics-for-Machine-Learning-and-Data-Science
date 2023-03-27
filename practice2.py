import numpy as np



A=np.array([
    [-1,3],
    [3,2]
],dtype=float)

b=np.array([7,1],dtype=float)

print("Matrix A:")
print(A)
print("----------")
print("Array b:")
print(b)
print("----------")
print(f"Shape of A: {A.shape}")
print(f"Shape of b: {b.shape}")
print("----------")

x=np.linalg.solve(A,b)
print(f"Solution: {x}")

d=np.linalg.det(A)

print(f"determinant of matrix A: {d:.2f}")

