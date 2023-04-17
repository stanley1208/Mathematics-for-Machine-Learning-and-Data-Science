import numpy as np

def MultiplyRow(M,row_num,row_num_multiple):
    M_new=M.copy()
    M_new[row_num]=M_new[row_num]*row_num_multiple
    return M_new

def AddRow(M,row_num_1,row_num_2,row_num_1_multiple):
    M_new=M.copy()
    M_new[row_num_2]=row_num_1_multiple*M_new[row_num_1]+M_new[row_num_2]
    return M_new

def SwapRows(M,row_num_1,row_num_2):
    M_new=M.copy()
    M_new[[row_num_1,row_num_2]]=M_new[[row_num_2,row_num_1]]
    return M_new


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

A_system=np.hstack((A,b.reshape(3,1)))
print(A_system)


print("Original matrix:")
print(A_system)
print("Matrix after MultiplyRow:")
print(MultiplyRow(A_system,2,2))

print("Original matrix:")
print(A_system)
print("Matrix after AddRow:")
print(AddRow(A_system,1,2,1/2))

print("Original matrix:")
print(A_system)
print("Matrix after SwapRows:")
print(SwapRows(A_system,0,2))