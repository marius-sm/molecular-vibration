# coding=utf-8
import numpy as np
import solve
import parameters
import plot_solution
import matplotlib.pyplot as plt
import gradient


X0 = np.zeros((parameters.n_atomes, 3))
min_LJ = 2**(1/6)
X0[1,0] = min_LJ
X0[2,0] = min_LJ/2
X0[2,1] = np.sqrt(0.75*min_LJ**2)

# offset
# X0[0,1] -= 0.1
# X0[1,1] -= 0.1
# X0[2,1] += 0.2



# on obtient le mode de vibration ou l'angle ne change pas
delta = 0.2
Delta = 2/3*delta*np.cos(np.pi/6)
X0[0,0] -= delta*np.sin(np.pi/6)
X0[0,1] -= delta*np.cos(np.pi/6) - Delta
X0[1,0] += delta*np.sin(np.pi/6)
X0[1,1] -= delta*np.cos(np.pi/6) - Delta
X0[2,1] += Delta


Y = solve.solve_ode(X0, 0, 100, parameters.n_frames)
plot_solution.plot(Y)

# On regarde la coordonnée x de l'atome 0
X = []
T = []
for t in range(len(Y)):
    T.append(t)
    X.append(Y[t][0][0])
plt.plot(T,X)
FX = np.fft.rfft(X)
plt.plot(T[:len(FX)],np.abs(FX))
plt.show()
