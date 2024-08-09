"""
@daniel.mimouni
This code aim at producing a reduced tree that approximate the better the initial large tree
using the Kovacevic and Pichler algorithm, improved with the MAM (or IBP) methods to gain speed.
Here I will study the influence of the strucutre of the initial large tree on the computation time of each
method (LP, MAM, MAM multiprocess, IBP).
The varying parameters are the number of nodes and the number of children per node at fixed stage t.
"""
from reduction import *
from application2 import *


# name = f'outputs/gr2000_ec150_mean_time_00.pkl'
# with open(name, 'rb') as f:
#     res = pickle.load(f)
# gr = 2000
# ec = 150
# plt.figure()
# plt.plot([i for i in range(1,gr+1) if i%ec==0 or i==1], res[0], '-*', label='LP')
# plt.plot([i for i in range(1,gr+1) if i%ec==0 or i==1], res[1], '-*', label='MAM')
# # plt.plot([i for i in range(1,gr+1) if i%ec==0 or i==1], res[2], '-*', label='IBP')
# plt.title('1 root and 2 nodes tree')
# plt.xlabel('number of children at last stage')
# plt.ylabel('computation time')
# plt.grid()
# plt.legend()
# plt.show()

# n = 3
# name = f'outputs/mean_time_{n}.pkl'
# with open(name, 'rb') as f:
#     res = pickle.load(f)
# gr = 600
# ec = 75
# plt.figure()
# plt.plot([i for i in range(1,gr+1) if i%ec==0 or i==1], res[0], '-*', label='LP')
# plt.plot([i for i in range(1,gr+1) if i%ec==0 or i==1], res[1], '-*', label='MAM')
# # plt.plot([i for i in range(1,gr+1) if i%ec==0 or i==1], res[2], '-*', label='IBP')
# plt.title(f'1 root and 2 then {n} nodes tree')
# plt.xlabel('number of children at last stage')
# plt.ylabel('computation time')
# plt.grid()
# plt.legend()
# # plt.show()

l_n = np.array([0,1,2,3,4,5,6,7,8,15,20,25]) # [0,1,2,3,5,8, 15, 20, 25]
LP = {}
MAM = {}
gr = 600
ec = 75
dim = gr // ec + 1
x_values = [i for i in range(1, gr + 1) if i % ec == 0 or i == 1]
plt.figure()
l_n2 = [0, 1,2,3,4] #,6,7,8,9,10] #,8, 15, 20, 25]
for n in l_n2:
    try:
        name = f'outputs/mean_time_{n}_v22.pkl'
        with open(name, 'rb') as f:
            res = pickle.load(f)
    except:
        name = f'outputs/mean_time_{n}.pkl'
        with open(name, 'rb') as f:
            res = pickle.load(f)

    if n==0:
        name = f'outputs/mean_time_{n}.pkl'
        with open(name, 'rb') as f:
            res = pickle.load(f)
        plt.plot(x_values, res[0][:dim], '--', color='tab:blue', label=f'LP')
        plt.annotate(f'n={n}', (x_values[-1], res[0][:dim][-1]), textcoords="offset points", xytext=(10, 0), ha='center')
        plt.plot(x_values, res[1][:dim], '-.', color='orange', label=f'MAM')
        plt.annotate(f'n={n}', (x_values[-1], res[1][:dim][-1]), textcoords="offset points", xytext=(10, 0), ha='center')

    if n != 0:
        plt.plot(x_values, res[0][:dim], '--', color='tab:blue')
        plt.annotate(f'n={2*n}', (x_values[-1], res[0][:dim][-1]), textcoords="offset points", xytext=(10, 0), ha='center')
        plt.plot(x_values, res[1][:dim], '-.', color='orange')
        plt.annotate(f'n={2*n}', (x_values[-1], res[1][:dim][-1]), textcoords="offset points", xytext=(10, 0), ha='center')
plt.title(f'1 root and 2 then n nodes tree')
plt.xlabel('|n+|')
plt.ylabel('computation time (s)')
plt.grid()
plt.legend()

# FIND THE TRESHOLD
if True:
    l_n2 = [ i for i in range(11)]
    treshold = []
    for n in l_n:
        try:
            name = f'outputs/mean_time_{n}_v2.pkl'
            with open(name, 'rb') as f:
                res = pickle.load(f)
        except:
            name = f'outputs/mean_time_{n}.pkl'
            with open(name, 'rb') as f:
                res = pickle.load(f)
        ll = np.array(res[1][:dim]) - np.array(res[0][:dim])
        i = next((i for i, x in enumerate(ll) if x < 0), None)
        treshold.append(x_values[i])
    plt.figure()
    plt.plot(l_n*2, treshold, '-o', color='orange')
    plt.title(f'Evolution of the treshold with n')
    plt.xlabel('n')
    plt.ylabel('treshold |n+|')
    plt.grid()
    plt.legend()


# PLOT SURFACE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

n_plus = np.array([i for i in range(1,gr+1) if i%ec==0 or i==1])
# l_n = [1,2,3]
LP = {}
MAM = {}
for n in l_n:
    try:
        name = f'outputs/mean_time_{n}_v2.pkl'
        with open(name, 'rb') as f:
            res = pickle.load(f)
    except:
        name = f'outputs/mean_time_{n}.pkl'
        with open(name, 'rb') as f:
            res = pickle.load(f)
    LP[n] = res[0][:dim]
    MAM[n] = res[1][:dim]

LPs = np.vstack([LP[n] for n in l_n])
MAMs = np.vstack([MAM[n] for n in l_n])

# Create the grid for the surface plots
X, Y = np.meshgrid(n_plus[:], l_n*2)

# Plot the surfaces
fig = plt.figure()

# Plot LP surface
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, LPs, cmap='Oranges', alpha=0.75) #viridis
ax.plot_surface(X, Y, MAMs, cmap='summer', alpha=.75) #magma
ax.set_title('LP vs MAM')
ax.set_xlabel('|n+|')
ax.set_ylabel('n')
ax.set_zlabel('computation time (s)')


plt.tight_layout()
plt.show()
