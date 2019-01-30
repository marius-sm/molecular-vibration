import numpy as np
import parameters

def dist(i, j, X):
    """
    Distance entre les atomes i et j
    X : un np.array de 3*n_atomes
    """
    return np.sqrt(float( (X[3*i]-X[3*j])**2 + (X[3*i+1]-X[3*j+1])**2 + (X[3*i+2]-X[3*j+2])**2 ))

def LJ(X):
    """
    Potentiel de Lennard Jones
    X : un np.array de 3*n_atomes
    Retourne un scalaire
    """
    V = 0
    n_atomes = int(len(X)/3)
    for i in range(n_atomes):
        for j in range(i):
            r = dist(i,j,X)
            V += 4*parameters.E[i,j]*( (parameters.d[i,j]/r)**12 - (parameters.d[i,j]/r)**6 )
    return V

def gradient_LJ_FD(X):
    """
    Gradient du potentiel de Lennard Jones
    X : un np.array de 3*n_atomes
    Retourne un np.array de 3*n_atomes
    """
    n_atomes = int(len(X)/3)
    approx = np.zeros(3*n_atomes)
    for k in range(3*n_atomes):
        v = np.zeros(3*n_atomes)
        h = 1e-8
        v[k] = 1
        approx[k] = (LJ(X+h*v) - LJ(X - h*v))/2/h
    return approx
