# PYTHON FILE
# GRID WORLD
# VALUE ITERATION CASE
# FUNCTIONS TO PLOT


import matplotlib.pyplot as plt
import matplotlib
import numpy as np

plt.rcParams['figure.figsize'] = [10, 7]
plt.rcParams['figure.dpi'] = 100 
plt.rcParams['font.size'] = 6

def plot_world(World):
    # ------------------
    Ly, Lx = World.shape

    fig, ax = plt.subplots()
    #fig.set_size_inches(4, 3) # i add this to decrease the size
    im = ax.imshow(World, cmap=plt.get_cmap("Spectral"))
    
    # We want to show all ticks...
    ax.set_xticks(np.arange(Lx))
    ax.set_yticks(np.arange(Ly))

    goal = np.where(np.logical_or( World > 0.0, World < -1.0))
    blocks = np.where(World == -1.0)
    # Loop over data dimensions and create text annotations.
    for i in range(Lx):
        for j in range(Ly):
            if np.logical_and(goal[0]==j,goal[1]==i).any():
                text = ax.text(i,j, 'G{}'.format(int(World[j,i])), ha="center", va="center", color="black")
            elif np.logical_and(blocks[0]==j,blocks[1]==i).any():
                 text = ax.text(i,j, 'X', ha="center", va="center", color="black", backgroundcolor="black")
            else:
                pass
    # -------------------

    

def plot_world_values(World, Values):
    # ------------------
    Ly, Lx = World.shape

    fig, (ax, ax2) = plt.subplots(1,2)
    im = ax.imshow(World, cmap=plt.get_cmap("Spectral"))

    # We want to show all ticks...
    ax.set_xticks(np.arange(Lx))
    ax.set_yticks(np.arange(Ly))

    goal = np.where(np.logical_or( World > 0.0, World < -1.0))
    blocks = np.where(World == -1.0)
    # Loop over data dimensions and create text annotations.
    for i in range(Lx):
        for j in range(Ly):
            if np.logical_and(goal[0]==j,goal[1]==i).any():
                text = ax.text(i,j, 'G{}'.format(World[j,i]), ha="center", va="center", color="black")
            elif np.logical_and(blocks[0]==j,blocks[1]==i).any():
                text = ax.text(i,j, 'X', ha="center", va="center", color="black", backgroundcolor="black")
            else:
                pass

    im2 = ax2.imshow(Values, cmap=plt.get_cmap("Spectral"))

    # We want to show all ticks...
    ax2.set_xticks(np.arange(Lx))
    ax2.set_yticks(np.arange(Ly))

    # Loop over data dimensions and create text annotations.
    for i in range(Lx):
        for j in range(Ly):
            if np.logical_and(goal[0]==j, goal[1]==i).any():
                text = ax2.text(i,j, 'G{}'.format(World[j,i]), ha="center", va="center", color="black")
            elif np.logical_and(blocks[0]==j,blocks[1]==i).any():
                text = ax2.text(i,j, 'X', ha="center", va="center", color="black", backgroundcolor="black")
            else:
                text = ax2.text(i, j, '{:.2f}'.format(Values[j, i]), ha="center", va="center", color="black")
                
    # -------------------

    

def plot_world_values_policy(World, Values, Policy):
    # ------------------
    Ly, Lx = World.shape

    fig, (ax, ax2, ax3) = plt.subplots(1,3)
    im = ax.imshow(World, cmap=plt.get_cmap("Spectral"))

    # We want to show all ticks...
    ax.set_xticks(np.arange(Lx))
    ax.set_yticks(np.arange(Ly))

    goal = np.where(np.logical_or( World > 0.0, World < -1.0))
    blocks = np.where(World == -1.0)
    # Loop over data dimensions and create text annotations.
    for i in range(Lx):
        for j in range(Ly):
            if np.logical_and(goal[0]==j,goal[1]==i).any():
                text = ax.text(i,j, 'G-{}'.format(World[j,i]), ha="center", va="center", color="black")
            elif np.logical_and(blocks[0]==j,blocks[1]==i).any():
                text = ax.text(i,j, 'X', ha="center", va="center", color="black", backgroundcolor="black")
            else:
                pass

    im2 = ax2.imshow(Values, cmap=plt.get_cmap("Spectral"))

    # We want to show all ticks...
    ax2.set_xticks(np.arange(Lx))
    ax2.set_yticks(np.arange(Ly))

    # Loop over data dimensions and create text annotations.
    for i in range(Lx):
        for j in range(Ly):
            if np.logical_and(goal[0]==j, goal[1]==i).any():
                text = ax2.text(i,j, 'G{}'.format(World[j,i]), ha="center", va="center", color="black")
                text = ax3.text(i,j, 'G{}'.format(World[j,i]), ha="center", va="center", color="black")
            elif np.logical_and(blocks[0]==j,blocks[1]==i).any():
                text = ax2.text(i,j, 'X', ha="center", va="center", color="black", backgroundcolor="black")
                text = ax3.text(i,j, 'X', ha="center", va="center", color="black", backgroundcolor="black")
            else:
                text = ax2.text(i, j, '{:.2f}'.format(Values[j, i]), ha="center", va="center", color="black")
    
    im3 = ax3.imshow(Values, cmap=plt.get_cmap("Spectral"))
    X = np.arange(Lx)
    Y = np.arange(Ly)
    U, V = Policy[:,:,1], -Policy[:,:,0]
    q = ax3.quiver(X, Y, U, V, color="black")

    # -------------------
    