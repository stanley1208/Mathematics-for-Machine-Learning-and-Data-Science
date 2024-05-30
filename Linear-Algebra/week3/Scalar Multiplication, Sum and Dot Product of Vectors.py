import numpy as np
import time

import matplotlib.pyplot as plt




def plot_vectors(list_v, list_label, list_color):
    _, ax = plt.subplots(figsize=(10, 10))
    ax.tick_params(axis='x', labelsize=14)
    ax.tick_params(axis='y', labelsize=14)
    ax.set_xticks(np.arange(-10, 10))

    ax.set_yticks(np.arange(-10, 10))

    plt.axis([-10, 10, -10, 10])

    for i, v in enumerate(list_v):
        sgn = 0.4 * np.array([[1] if i == 0 else [i] for i in np.sign(v)])
        plt.quiver(v[0], v[1], color=list_color[i], angles='xy', scale_units='xy', scale=1)
        ax.text(v[0] - 0.2 + sgn[0], v[1] - 0.2 + sgn[1], list_label[i], fontsize=14, color=list_color[i])

    plt.grid()
    plt.show()


v = np.array([
    [1],
    [3]
])
# Arguments: list of vectors as NumPy arrays, labels, colors.
plot_vectors([v], [f"$v$"], ["black"])
plot_vectors([v, 2*v, -2*v], [f"$v$", f"$2v$", f"$-2v$"], ["black", "green", "blue"])


# Sum of Vectors
v=np.array([[1],[3]])
w=np.array([[4],[-1]])

plot_vectors([v,w,v+w],[f'$v$',f'$w$',f'$v+w$'],['black','red','green'])

# Norm of a Vector
print("The norm of vector v is",np.linalg.norm(v))


# Dot Product
x=[1,-2,-5]
y=[4,3,-1]

def dot(x,y):
    s=0
    for xi,yi in zip(x,y):
        s+=xi*yi
    return s

print("The dot product of x and y:", dot(x,y))
print("Numpy dot product of x and y:", np.dot(x,y))

print("The @ dot product of x and y:", np.array(x) @ np.array(y))

try:
    print(x @ y)
except TypeError as err:
    print(err)

x=np.array(x)
y=np.array(y)


a=np.random.rand(1000000)
b=np.random.rand(1000000)


t1=time.time()
c=dot(a,b)
t2=time.time()
print('Dot product: ',c)
print('Time for dot product: ',str(t2-t1)+'seconds')

t1=time.time()
c=np.dot(a,b)
t2=time.time()
print('numpy Dot product: ',c)
print('Time for numpy dot product: ',str(t2-t1)+'seconds')



t1=time.time()
c=a@b
t2=time.time()
print('@ Dot product: ',c)
print('Time for @ dot product: ',str(t2-t1)+'seconds')


i=np.array([1,0,0])
j=np.array([0,1,0])
print("Dot product: ",dot(i,j))