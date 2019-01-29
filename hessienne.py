import numpy as np
import parameters

def f1(E,d,r):
    return(24*E*(d**6/r**7-2*d**12/r**13))

def f2(E,d,r):
    return(24*(26*d**12/4**14-7*d**6/r**8))

X = np.zeros((parameters.n_atomes, 3))
X = np.floor(10*np.random.random((parameters.n_atomes, 3)))


def M(i,j):
    M=np.zeros((parameters.n_atomes*3,parameters.n_atomes*3))
    M[3*i:3*i+3,3*i:3*i+3]=np.dot(np.transpose(X[i]),X[i])
    M[3*i:3*i+3,3*j:3*j+3]=np.dot(np.transpose(X[i]),X[j])
    M[3*j:3*j+3,3*i:3*i+3]=np.dot(np.transpose(X[j]),X[i])
    M[3*j:3*j+3,3*j:3*j+3]=np.dot(np.transpose(X[j]),X[j])
    return M

print(M(0,2))

def N(i,j):
    N=np.zeros((parameters.n_atomes*3,parameters.n_atomes*3))
    N[3*i:3*i+3,3*i:3*i+3]=2*np.eye(3)
    N[3*i:3*i+3,3*j:3*j+3]=-2*np.eye(3)
    N[3*j:3*j+3,3*i:3*i+3]=-2*np.eye(3)
    N[3*j:3*j+3,3*j:3*j+3]=2*np.eye(3)
    return N

print(N(0,2))

def r(i,j):
    return np.linalg.norm(X[i]-X[j])

def H(i,j):
    return 4*f2(parameters.E[i,j], parameters.d[i,j], r(i,j))*M(i,j)+f1(parameters.E[i,j], parameters.d[i,j], r(i,j))*N(i,j)

print(np.floor(1000*H(0,2)))

def hessienne():
    h=np.zeros((parameters.n_atomes*3,parameters.n_atomes*3))
    for j in range(parameters.n_atomes):
        for i in range(j):
            h+=H(i,j)
    return h

print(np.floor(1000*hessienne()))
