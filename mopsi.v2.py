# coding=utf-8
import numpy as np
import solve
import parameters
import plot_solution
import matplotlib.pyplot as plt
import gradient


X0 = np.zeros((parameters.n_atomes, 3))
min_LJ = 2**(1/6)
X0[0,0] = 0
X0[1,0] = min_LJ
X0[2,0] = min_LJ/2
X0[2,1] = np.sqrt(0.75*min_LJ**2)

# offset
#X0[0,1] -= 0.1
#X0[1,1] -= 0.1
#X0[2,1] += 0.2

# X = []
# Y = []
# G = []
# delta = 0.01
# for n in range(100):
#     X0[0,0] -= delta
#     X0[1,0] += delta
#     X.append(X0[1,0] - X0[0,0])
#     Y.append(gradient.LJ(np.reshape(X0,(1, 3*parameters.n_atomes))[0]))
#     G.append(gradient.gradient(np.reshape(X0,(1, 3*parameters.n_atomes))[0]))
# plt.plot(X,Y)
# plt.plot(X,G)
# plt.show()



# on obtient le mode de vibration ou l'angle ne change pas
delta = 0.1
Delta = 2/3*delta*np.cos(np.pi/6)
X0[0,0] -= delta*np.sin(np.pi/6)
X0[0,1] -= delta*np.cos(np.pi/6) - Delta
X0[1,0] += delta*np.sin(np.pi/6)
X0[1,1] -= delta*np.cos(np.pi/6) - Delta
X0[2,1] += Delta

#X0 = np.reshape(X0,(1, 3*parameters.n_atomes))[0]
v = np.random.rand(1, 3*parameters.n_atomes)[0]
v = v/np.linalg.norm(v)
h = 1e-8

#print( (gradient.LJ(X0 + h*v) - gradient.LJ(X0))/h)
#print( np.dot(gradient.gradient(X0),v) )


Y = solve.solve_ode(X0, 0, 10, parameters.n_frames)
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
