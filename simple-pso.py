##  Author: Heitor Teixeira de Azambuja
##  Date:   07/03/2022

import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.animation as anim

MAX_VEL = .4
ITERATIONS = 10
x, y = [], []
minx = None
miny = None
N = 100
inertia = 1  
personalWeight = 2
socialWeight = 2
vel = []
pBest = []


def fun(x, y):
    return x**2 + y**2  # Parabole
    # return - x**2 - y**2  # Inverse parabole
    # return -x**2 + y**2


#  Figure Creation and initial plots
fig = plt.figure(num='3D Parabola PSO')
fig.set_size_inches(11, 6)

fig.suptitle('3D Parabola Particle swarm optimization', fontsize=14, fontweight='bold')
ax = fig.add_subplot(121, projection='3d', title='Mesh')
x = y = np.arange(-4.0, 4.0, 0.05)
X, Y = np.meshgrid(x, y)
zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)

ax.plot_surface(X, Y, Z, cmap=cm.coolwarm)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax = fig.add_subplot(122, title='Z heat map')
ax = plt.imshow(Z, cmap=cm.coolwarm, extent=[-4,4,-4,4])
plt.colorbar(ax)
scat = plt.scatter(x,y, c='black', s=15)
scatMin = plt.scatter(x,y, c='r', s=15)


#  Initialize particles animation
def initAnim():
    print('Initializing particles - ', end=' ')
    global x, y, scat, scatMin, scat, scatMin, vel, minx, miny
    x, y = [], []
    
    scat.remove()
    scatMin.remove()
    
    x.append(np.random.rand(N) * 8 - 4)
    y.append(np.random.rand(N) * 8 - 4)
    
    minimum = 999999999999
    minx = None
    miny = None
    for i in range(len(x[0])):
        val = fun(x[0][i], y[0][i])
        vel.append(np.random.rand(2) * MAX_VEL * 2 - MAX_VEL)
        pBest.append(np.array([x[0][i], y[0][i]]))
        if val < minimum:
            minimum = val
            minx = x[0][i]
            miny = y[0][i]

    print('Global minimum: ' + str(minimum))

    scat = plt.scatter(x,y, c='black', s=15)
    scatMin = plt.scatter(minx,miny, c='r', s=15)

    plt.pause(.3)


#  Compute swarm particle optimization
def animate(i):
    print ('Iteration ' + str(i + 1) + ' - ', end=' ')
    global x, y, minx, miny, scat, scatMin, vel, inertia, personalWeight, socialWeight
    minimum = 999999999999
    
    for i in range(len(x[0])):
        # compute new velocity
        vel[i] = (inertia * vel[i]) + (personalWeight * np.random.rand() * (pBest[i] - np.array([x[0][i], y[0][i]]))) + (socialWeight * np.random.rand() * ([minx, miny] - np.array([x[0][i], y[0][i]])))
        # max velocity limit
        vel[i][0] = min(vel[i][0], MAX_VEL)
        vel[i][1] = min(vel[i][1], MAX_VEL)
        # min velocity limit
        vel[i][0] = max(vel[i][0], -MAX_VEL)
        vel[i][1] = max(vel[i][1], -MAX_VEL)
        # compute new position
        x[0][i] = x[0][i] + vel[i][0]
        y[0][i] = y[0][i] + vel[i][1]
        # check if new position is inside the domain
        if x[0][i] > 4:
            x[0][i] = 4
            vel[i][0] = -vel[i][0]
        if x[0][i] < -4:
            x[0][i] = -4
            vel[i][0] = -vel[i][0]
        if y[0][i] > 4:
            y[0][i] = 4
            vel[i][1] = -vel[i][1]
        if y[0][i] < -4:
            y[0][i] = -4
            vel[i][1] = -vel[i][1]
        # update pBest
        val = fun(x[0][i], y[0][i])
        if val < fun(pBest[i][0], pBest[i][1]):
            pBest[i] = [x[0][i], y[0][i]]
        # update min
        if val < minimum:
            minimum = val
            minx = x[0][i]
            miny = y[0][i]
    
    print('Global minimum: ' + str(minimum))

    scat.remove()
    scatMin.remove()
    scat = plt.scatter(x,y, c='black', s=15)
    scatMin = plt.scatter(minx,miny, c='r', s=15)


ani = anim.FuncAnimation(fig, animate, frames=ITERATIONS, init_func=initAnim, interval=500, repeat=True) 

plt.tight_layout(w_pad=4)
plt.show()