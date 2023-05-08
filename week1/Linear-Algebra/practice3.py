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


# ref is an abbreviation of the row echelon form
A_ref=SwapRows(A_system,0,2)
print(A_ref)

A_ref=AddRow(A_ref,0,1,2)
print(A_ref)

A_ref=AddRow(A_ref,0,2,4)
print(A_ref)

A_ref=AddRow(A_ref,1,2,-1)
print(A_ref)

A_ref=MultiplyRow(A_ref,2,-1/12)
print(A_ref)

x_3=-2
x_2=(A_ref[1,3]-A_ref[1,2]*x_3)/A_ref[1,1]
x_1=(A_ref[0,3]-A_ref[0,2]*x_3-A_ref[0,1]*x_2)/A_ref[0,0]

print(x_1,x_2,x_3)


A_2=np.array([
    [1,1,1],
    [0,1,-3],
    [2,1,5]
],dtype=float)

b_2=np.array([2,1,0],dtype=float)

d_2=np.linalg.det(A_2)

print(f"Determinant of matrix A_2 is: {d_2:.2f}")

# numpy.linalg.LinAlgError: Singular matrix
# x=np.linalg.solve(A_2,b_2)
# print(x)

A_2_system=np.hstack((A_2,b_2.reshape(3,1)))
print(A_2_system)

A_ref=AddRow(A_2_system,0,2,-2)
print(A_ref)

A_ref=AddRow(A_ref,1,2,1)
print(A_ref)


A_3=np.array([
    [1,1,1],
    [0,1,-3],
    [2,1,5]
],dtype=float)

b_3=np.array([2,1,3],dtype=float)

d_3=np.linalg.det(A_3)

print(f"Determinant of matrix A_3 is: {d_3:.2f}")

A_3_system=np.hstack((A_3,b_3.reshape(3,1)))
print(A_3_system)

A_ref=AddRow(A_3_system,0,2,-2)
print(A_ref)

A_ref=AddRow(A_ref,1,2,1)
print(A_ref)

