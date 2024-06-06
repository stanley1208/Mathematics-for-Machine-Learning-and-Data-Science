import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse.linalg
import utils
import w4_unittest


A = np.array([[2, 3],[2, 1]])
e1 = np.array([[1],[0]])
e2 = np.array([[0],[1]])

# utils.plot_transformation(A,e1,e2,vector_name='e');


A_eig = np.linalg.eig(A)

print("\n")

print(f"Matrix A:\n{A} \n\nEigenvalues of matrix A:\n{A_eig[0]}\n\nEigenvectors of matrix A:\n{A_eig[1]}")
utils.plot_transformation(A, A_eig[1][:,0], A_eig[1][:,1]);


# define transformation matrix as a numpy array
A_reflection_yaxis=np.array([[-1,0],[0,1]])
# find the eigenvalues and eigenvectors of matrix
A_reflection_yaxis_eig=np.linalg.eig(A_reflection_yaxis)

print(f'Matrix A_reflection_yaxis:\n{A_reflection_yaxis}')
print(f'Eigenvalues {A_reflection_yaxis_eig[0]}')
print(f'Eigenvectors {A_reflection_yaxis_eig[1]}')
utils.plot_transformation(A_reflection_yaxis, A_reflection_yaxis_eig[1][:,0], A_reflection_yaxis_eig[1][:,1]);