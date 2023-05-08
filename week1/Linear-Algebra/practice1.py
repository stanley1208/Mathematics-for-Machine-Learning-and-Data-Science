import numpy as np


# Run the "Hello World" in the cell below to print "Hello World".
test = "Hello World"
print(test)


one_dimensional_arr=np.array([10,12])
print(one_dimensional_arr)

# Create and print a NumPy array 'a' containing the elements 1, 2, 3.
a=np.array([1,2,3])
print(a)

# Create an array with 3 integers, starting from the default integer 0.
b=np.arange(3)
print(b)

# Create an array that starts from the integer 1, ends at 20, incremented by 3.
c=np.arange(1,20,3)
print(c)

lin_spaces_arr=np.linspace(0,100,5)
print(lin_spaces_arr)

lin_spaces_arr_int=np.linspace(0,100,5,dtype=int)
print(lin_spaces_arr_int)

c_int=np.arange(1,20,3,dtype=int)
print(c_int)

b_float=np.arange(3,dtype=float)
print(b_float)

char_arr=np.array(['Welcome to Math for Machine Learning!'])
print(char_arr)
print(char_arr.dtype)

# Return a new array of shape 3, filled with ones.
ones_arr=np.ones(3)
print(ones_arr)

# Return a new array of shape 3, filled with zeroes.
zeros_arr=np.zeros(3)
print(zeros_arr)

# Return a new array of shape 3, without initializing entries.
empt_arr=np.empty(3)
print(empt_arr)

# Return a new array of shape 3 with random numbers between 0 and 1.
rand_arr=np.random.rand(3)
print(rand_arr)


# Create a 2 dimensional array (2-D)
two_dim_arr=np.array([[1,2,3],[4,5,6]])
print(two_dim_arr)

# 1-D array
one_dim_arr=np.array([1,2,3,4,5,6])

# Multidimensional array using reshape()
multi_dim_arr=np.reshape(one_dim_arr    # the array to be reshaped
                         ,(2,3)     # dimensions of the new array
                         )
print(multi_dim_arr)

# Dimension of the 2-D array multi_dim_arr
print(multi_dim_arr.ndim)
# Shape of the 2-D array multi_dim_arr
# Returns shape of 2 rows and 3 columns
print(multi_dim_arr.shape)
# Size of the array multi_dim_arr
# Returns total number of elements
print(multi_dim_arr.size)


arr_1=np.array([2,4,6])
arr_2=np.array([1,3,5])

# Adding two 1-D arrays
addition=arr_1+arr_2
print(addition)

# Subtracting two 1-D arrays
subtraction=arr_1-arr_2
print(subtraction)

# Multiplying two 1-D arrays elementwise
multiplication=arr_1*arr_2
print(multiplication)

# broadcasting
vector=np.array([1,2])
vector=vector*1.6
print(vector)

# Select the third element of the array. Remember the counting starts from 0.
a=[1,2,3,4,5]
print(a[2])

# Select the first element of the array.
print(a[0])

# Indexing on a 2-D array
two_dim=np.array(([1,2,3],
                  [4,5,6],
                  [7,8,9]))

# Select element number 8 from the 2-D array using indices i, j.
print(two_dim[2][1])


# Slice the array a to get the array [2,3,4]
sliced_arr=a[1:4]
print(sliced_arr)
# Slice the array a to get the array [1,2,3]
sliced_arr=a[:3]
print(sliced_arr)
# Slice the array a to get the array [3,4,5]
sliced_arr=a[2:]
print(sliced_arr)
# Slice the array a to get the array [1,3,5]
sliced_arr=a[::2]
print(sliced_arr)

# Note that a == a[:] == a[::]
print(a == a[:] == a[::])

# Slice the two_dim array to get the first two rows
sliced_arr_1=two_dim[0:2]
print(sliced_arr_1)
# Similarily, slice the two_dim array to get the last two rows
sliced_arr_1=two_dim[1:]
print(sliced_arr_1)

sliced_two_dim_cols=two_dim[:,1]
print(sliced_two_dim_cols)


a1=np.array([[1,1],[2,2]])
a2=np.array([[3,3],[4,4]])

print("a1:",a1)
print("a2:",a2)

# Stack the arrays vertically
vert_stack=np.vstack((a1,a2))
print(vert_stack)
# Stack the arrays horizontally
horz_stack=np.hstack((a1,a2))
print(horz_stack)



zeros_arr=np.zeros(3)
print(zeros_arr)


empt_arr=np.empty(3)
print(empt_arr)

print(zeros_arr==empt_arr)