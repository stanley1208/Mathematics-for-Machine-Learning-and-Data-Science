import numpy as np



A=np.array([
    [4,-3,1],
    [2,1,3],
    [-1,2,-5]
],dtype=float)

b=np.array([-10,0,17],dtype=float)

print("Matrix A")
print(A)
print("Array b")
print(b)

print(f"Shape of A:{A.shape}")
print(f"Shape of b:{b.shape}")

x=np.linalg.solve(A,b)
print(f"Solution: {x}")

d=np.linalg.det(A)
print(f"Determinant of matrix A is: {d:.2f}")
