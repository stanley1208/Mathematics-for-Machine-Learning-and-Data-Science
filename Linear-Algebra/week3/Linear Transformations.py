import numpy as np
import cv2

def T(v):
    w=np.zeros((3,1))
    w[0,0]=3*v[0,0]
    w[2,0]=-2*v[1,0]

    return w

v=np.array([[3],[5]])
w=T(v)

print("Original vector:\n",v,"\nResult of the transformation:\n",w)


print()

u=np.array([[1],[-2]])
v=np.array([[2],[4]])

k=10

print("T(k*v):\n",T(k*v),"\n k*T(v):\n",k*T(v),"\n\n")
print("T(u+v):\n",T(u+v),"\n T(u)+T(v):\n",T(u)+T(v),"\n\n")


def L(v):
    A=np.array([[3,0],[0,0],[0,-2]])
    print("Transformation matrix:\n",A,"\n")
    w=A@v

    return w

v=np.array([[3],[5]])
w=L(v)

print("Original vector:\n",v,"\n Result of the transformation:\n",w)



