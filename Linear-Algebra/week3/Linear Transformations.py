import matplotlib.pyplot as plt
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


def T_hscaling(v):
    A=np.array([[2,0],[0,1]])
    w=A@v
    return w

def transform_vectors(T,v1,v2):
    V=np.hstack((v1,v2))
    W=T(V)
    return W

e1=np.array([[1],[0]])
e2=np.array([[0],[1]])


transformation_result_hscaling=transform_vectors(T_hscaling,e1,e2)
print(e1)
print(e2)
print(transformation_result_hscaling)


def T_reflection_yaxis(v):
    A=np.array([[-1,0],[0,1]])
    w=A@v
    return w

transformation_result_reflection_yaxis=transform_vectors(T_reflection_yaxis,e1,e2)
print(e1)
print(e2)
print(transformation_result_reflection_yaxis)


M_rotation_90_clockwise=np.array([[0,1],[-1,0]])
M_shear_x=np.array([[1,0.5],[0,1]])

print("M_rotation_90_clockwise:",M_rotation_90_clockwise)
print("M_shear_x:",M_shear_x)

print("M_rotation_90_clockwise@M_shear_x",M_rotation_90_clockwise@M_shear_x)
print("M_shear_x@M_rotation_90_clockwise",M_shear_x@M_rotation_90_clockwise)

