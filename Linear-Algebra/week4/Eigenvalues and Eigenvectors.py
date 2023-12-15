import numpy as np
import matplotlib.pyplot as plt
import w4_unittest




A=np.array([[2,3],[2,1]])
e1=np.array([[1],[0]])
e2=np.array([[0],[1]])


def plot_transformation(T, v1, v2):
    color_original = "#129cab"
    color_transformed = "#cc8933"

    _, ax = plt.subplots(figsize=(7, 7))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-6, 6))
    ax.set_yticks(np.arange(-6, 6))

    plt.axis([-6, 6, -6, 6])
    plt.quiver([0, 0], [0, 0], [v1[0], v2[0]], [v1[1], v2[1]], color=color_original, angles='xy', scale_units='xy',
               scale=1)
    plt.plot([0, v2[0], v1[0] + v2[0], v1[0]],
             [0, v2[1], v1[1] + v2[1], v1[1]],
             color=color_original)
    v1_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(v1)])
    ax.text(v1[0] - 0.2 + v1_sgn[0], v1[1] - 0.2 + v1_sgn[1], f'$v_1$', fontsize=14, color=color_original)
    v2_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(v2)])
    ax.text(v2[0] - 0.2 + v2_sgn[0], v2[1] - 0.2 + v2_sgn[1], f'$v_2$', fontsize=14, color=color_original)

    v1_transformed = T @ v1
    v2_transformed = T @ v2

    plt.quiver([0, 0], [0, 0], [v1_transformed[0], v2_transformed[0]], [v1_transformed[1], v2_transformed[1]],
               color=color_transformed, angles='xy', scale_units='xy', scale=1)
    plt.plot([0, v2_transformed[0], v1_transformed[0] + v2_transformed[0], v1_transformed[0]],
             [0, v2_transformed[1], v1_transformed[1] + v2_transformed[1], v1_transformed[1]],
             color=color_transformed)
    v1_transformed_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(v1_transformed)])
    ax.text(v1_transformed[0] - 0.2 + v1_transformed_sgn[0], v1_transformed[1] - v1_transformed_sgn[1],
            f'$T(v_1)$', fontsize=14, color=color_transformed)
    v2_transformed_sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(v2_transformed)])
    ax.text(v2_transformed[0] - 0.2 + v2_transformed_sgn[0], v2_transformed[1] - v2_transformed_sgn[1],
            f'$T(v_2)$', fontsize=14, color=color_transformed)

    plt.gca().set_aspect("equal")
    plt.show()


plot_transformation(A, e1, e2)

A_eig=np.linalg.eig(A)
print("Matrix A:",A)
print("Eigenvalues and eigenvectors of matrix",A_eig)
plot_transformation(A,A_eig[1][:,0],A_eig[1][:,1])
print(np.linalg.norm(A_eig[1][:,0]))
print(np.linalg.norm(A_eig[1][:,1]))


### START CODE HERE ###
# Define transformation matrix A_reflection_yaxis as a numpy array.
A_reflection_yaxis = np.array([[-1,0],[0,1]])
# Find eigenvalues and eigenvectors of matrix A_reflection_yaxis.
A_reflection_yaxis_eig = np.linalg.eig(A_reflection_yaxis)
### END CODE HERE ###

print("Matrix A_reflection_yaxis:\n", A_reflection_yaxis,
      "\n\n Eigenvalues and eigenvectors of matrix A_reflection_yaxis:\n", A_reflection_yaxis_eig)
plot_transformation(A_reflection_yaxis, A_reflection_yaxis_eig[1][:,0], A_reflection_yaxis_eig[1][:,1])
