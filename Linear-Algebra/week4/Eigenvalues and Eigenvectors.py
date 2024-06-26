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


# Shear in x-direction
# define transformation matrix A_shear_x
A_shear_x=np.array([[1,0.5],[0,1]])
# find the eigenvalues and eigenvectors of matrix
A_shear_x_eig=np.linalg.eig(A_shear_x)
print(f'Matrix A_shear_x:{A_shear_x}')
print(f'Eigenvalues of A_shear_x:{A_shear_x_eig[0]}')
print(f'Eigenvectors of A_shear_x:{A_shear_x_eig[1]}')
utils.plot_transformation(A_shear_x, A_shear_x_eig[1][:,0], A_shear_x_eig[1][:,1]);

A_rotation=np.array([[0,1],[-1,0]])
A_rotation_eig=np.linalg.eig(A_rotation)
print(f'Matrix A_shear_x:{A_rotation}')
print(f'Eigenvalues of A_shear_x:{A_rotation_eig[0]}')
print(f'Eigenvectors of A_shear_x:{A_rotation_eig[1]}')


A_identity=np.array([[1,0],[0,1]])
A_identity_eig=np.linalg.eig(A_identity)

utils.plot_transformation(A_identity,A_identity_eig[1][:,0], A_identity_eig[1][:,1]);

print(f'Matrix A_identity:{A_identity}')
print(f'Eigenvalues of A_identity:{A_identity_eig[0]}')
print(f'Eigenvectors of A_identity:{A_identity_eig[1]}')


A_scaling=np.array([[2,0],[0,2]])
A_scaling_eig=np.linalg.eig(A_scaling)

utils.plot_transformation(A_scaling,A_scaling_eig[1][:,0], A_scaling_eig[1][:,1]);

print(f'Matrix A_scaling:{A_scaling}')
print(f'Eigenvalues of A_scaling:{A_scaling_eig[0]}')
print(f'Eigenvectors of A_scaling:{A_scaling_eig[1]}')