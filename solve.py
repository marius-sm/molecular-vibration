import numpy as np
from scipy.integrate import odeint
from gradient import gradient

def solve_ode(X0, t0, tf, n_intervals):
    """
    X0 : un np.array de shape (n_atomes, 3)
    t0 : instant initial
    tf : instant final
    Retourne une liste y de shape (n_intervals, n_atomes, 3) dont y[T][n][j] donne la coordonnée j du n-ième atome à l'instant T
    """
    n_atomes = X0.shape[0]
    #print(n_atomes)
    X0 = np.reshape(X0,(1, 3*n_atomes))[0]
    Y0 = np.hstack((X0, np.zeros((1, 3*n_atomes))[0])) # Y0 = [x0,y0,z0,...,xn,yn,zn, vx0,vy0,vz0,...,vxn,vyn,vzn]

    T = np.linspace(t0,tf,n_intervals)
    def ode(Y,t):
        """
        Y: un np.array de 2*3*n_atomes de la forme [ x0,y0,z0,...,xn,yn,zn,    vx0,vy0,vz0,...,vxn,vyn,vzn ]
        """
        X, X_point = Y[:3*n_atomes], Y[3*n_atomes:]
        return np.hstack((X_point, -gradient(X)))
    Y = odeint(ode, Y0, T)
    Y = Y[:, :int(len(Y[0])/2)] # on ne garde que la position (et pas la vitesse)
    Y = np.reshape(Y, (n_atomes*n_intervals, 3))
    Y = np.vsplit(Y, n_intervals)
    return Y
