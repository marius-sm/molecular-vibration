# coding=utf-8
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

def gradient_3(i, j, k, X):
    """
    Dérivée partielle du potentielle entre les atomes i et j par rapport à la coordonnée k de X
    X : un np.array de 3*n_atomes
    Retourne un scalaire
    """
    n_atomes = int(len(X)/3)
    if not(3*i <= k <= 3*i+2 or 3*j <= k <= 3*j+2):
        return 0
    r = dist(i,j,X)
    if 3*i <= k <= 3*i+2:
        return 12*parameters.E[i,j]*(parameters.d[i,j]**6/r**7 - 2*parameters.d[i,j]**12/r**13) * 2*float(X[k] - X[3*j+k%3])
    else:
        return 12*parameters.E[i,j]*(parameters.d[i,j]**6/r**7 - 2*parameters.d[i,j]**12/r**13) * 2*float(X[k] - X[3*i+k%3])

def gradient_2(i, j, X):
    """
    Gradient du potentielle entre les atomes i et j
    X : un np.array de 3*n_atomes
    Retourne un np.array de 3*n_atomes
    """
    n_atomes = int(len(X)/3)
    g = np.zeros((1, 3*n_atomes))[0]
    for k in range(3*n_atomes):
        g[k] = gradient_3(i,j,k,X)
    return g

def gradient(X):
    """
    Gradient du potentiel
    X : un np.array de 3*n_atomes
    Retourne un np.array de 3*n_atomes
    """
    n_atomes = int(len(X)/3)
    g = np.zeros((1, 3*n_atomes))[0]
    for i in range(n_atomes):
        for j in range(i):
            g += gradient_2(i,j,X) #On divise par 10 par ce que sinon la solution part en couilles
    return g
