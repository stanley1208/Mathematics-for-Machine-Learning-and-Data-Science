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


### START CODE HERE ###
def MultiplyRow(M, row_num, row_num_multiple):
    # .copy() function is required here to keep the original matrix without any changes
    M_new = M.copy()
    # exchange row_num of the matrix M_new with its multiple by row_num_multiple
    # Note: for simplicity, you can drop check if  row_num_multiple has non-zero value, which makes the operation valid
    M_new[row_num] = M_new[row_num]*row_num_multiple
    return M_new


def AddRows(M, row_num_1, row_num_2, row_num_1_multiple):
    M_new = M.copy()
    # multiply row_num_1 by row_num_1_multiple and add it to the row_num_2,
    # exchanging row_num_2 of the matrix M_new with the result
    M_new[row_num_2] = M_new[row_num_1]*row_num_1_multiple+M_new[row_num_2]
    return M_new


def SwapRows(M, row_num_1, row_num_2):
    M_new = M.copy()
    # exchange row_num_1 and row_num_2 of the matrix M_new
    M_new[[row_num_1,row_num_2]] = M_new[[row_num_2,row_num_1]]
    return M_new
### END CODE HERE ###



A_test = np.array([
        [1, -2, 3, -4],
        [-5, 6, -7, 8],
        [-4, 3, -2, 1],
        [8, -7, 6, -5]
    ], dtype=np.dtype(float))
print("Original matrix:")
print(A_test)

print("\nOriginal matrix after its third row is multiplied by -2:")
print(MultiplyRow(A_test,2,-2))

print("\nOriginal matrix after exchange of the third row with the sum of itself and first row multiplied by 4:")
print(AddRows(A_test,0,2,4))

print("\nOriginal matrix after exchange of its first and third rows:")
print(SwapRows(A_test,0,2))


def augmented_to_ref(A, b):
    ### START CODE HERE ###
    # stack horizontally matrix A and vector b, which needs to be reshaped as a vector (4, 1)
    A_system = np.hstack((A,b.reshape((4,1))))

    # swap row 0 and row 1 of matrix A_system (remember that indexing in NumPy array starts from 0)
    A_ref = SwapRows(A_system,0,1)

    # multiply row 0 of the new matrix A_ref by -2 and add it to the row 1
    A_ref = AddRows(A_ref,0,1,-2)

    # add row 0 of the new matrix A_ref to the row 2, replacing row 2
    A_ref = AddRows(A_ref,0,2,1)

    # multiply row 0 of the new matrix A_ref by -1 and add it to the row 3
    A_ref = AddRows(A_ref,0,3,-1)

    # add row 2 of the new matrix A_ref to the row 3, replacing row 3
    A_ref = AddRows(A_ref,2,3,1)

    # swap row 1 and 3 of the new matrix A_ref
    A_ref = SwapRows(A_ref,1,3)

    # add row 2 of the new matrix A_ref to the row 3, replacing row 3
    A_ref = AddRows(A_ref,2,3,1)

    # multiply row 1 of the new matrix A_ref by -4 and add it to the row 2
    A_ref = AddRows(A_ref,1,2,-4)

    # add row 1 of the new matrix A_ref to the row 3, replacing row 3
    A_ref = AddRows(A_ref,1,3,1)

    # multiply row 3 of the new matrix A_ref by 2 and add it to the row 2
    A_ref = AddRows(A_ref,3,2,2)

    # multiply row 2 of the new matrix A_ref by -8 and add it to the row 3
    A_ref = AddRows(A_ref,2,3,-8)

    # multiply row 3 of the new matrix A_ref by -1/17
    A_ref = MultiplyRow(A_ref,3,-1/17)
    ### END CODE HERE ###

    return A_ref


A_ref = augmented_to_ref(A, b)

print(A_ref)