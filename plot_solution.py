# coding=utf-8
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from matplotlib import animation
import parameters
import numpy as np

def plot(y):
    """
    y : un np.array de shape (n_intervals, n_atomes, 3) dont y[T][n][j] donne la coordonnée j du n-ième atome à l'instant T
    """
    n_frames = len(y)
    n_atomes = len(y[0])
    t_ralonge = np.array([np.ones(n_atomes)*i for i in range(n_frames)]).flatten()
    y = np.concatenate((y))
    df = pd.DataFrame({"time": t_ralonge ,"x" : y[:,0], "y" : y[:,1], "z" : y[:,2]})
    def update_graph(num):
        data=df[df['time']==num]
        graph._offsets3d = (data.x, data.y, data.z)
        title.set_text('Atome oscillant dans le potentiel de LJ, time={}'.format(num))
        return title, graph


    fig = plt.figure()
    ax = fig.add_subplot(1,1,1, projection='3d')
    #ax.view_init(azim=0, elev=90)
    title = ax.set_title('Atome oscillant dans le potentiel de LJ')
    box_size = 2
    ax.set_xlim3d([-box_size, box_size])
    ax.set_xlabel('X')
    ax.set_ylim3d([-box_size, box_size])
    ax.set_ylabel('Y')
    ax.set_zlim3d([-box_size, box_size])
    ax.set_zlabel('Z')

    data=df[df['time']==0]
    graph = ax.scatter(data.x, data.y, data.z, marker="o")

    ani = animation.FuncAnimation(fig, update_graph, n_frames,
                                   interval=30, blit=False)

    plt.show()
