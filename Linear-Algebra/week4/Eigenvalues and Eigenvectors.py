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
