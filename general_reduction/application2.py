"""
@daniel.mimouni
This code aim at producing a reduced tree that approximate the better the initial large tree
using the Kovacevic and Pichler algorithm, improved with the MAM (or IBP) methods to gain speed.
Here I will study the influence of the strucutre of the initial large tree on the computation time of each
method (LP, MAM, MAM multiprocess, IBP).
The varying parameters are the number of nodes and the number of children per node at fixed stage t.
"""
# General
import sys

import matplotlib.pyplot as plt
from mpi4py import MPI
# My codes
from reduction_tree.tree_reduction_MPI import *
from reduction_tree.visualization_tree import *
from reduction import *

def make_growing_tree(T, arrangement = []):
    G = nx.DiGraph()
    i = 0
    ancestors = [0]
    G.add_node(0)
    G.nodes[i]['quantizer'] = random.randint(1, 20) - 10
    G.nodes[i]['stage'] = 0
    i = 1
    for t in range(1,T):
        nb = t
        if t<3:
            nb = 2
        if type(arrangement) == list:
            nb = arrangement[t]
        probabilities = np.random.random(nb)  # np.ones(len(children))
        probabilities /= np.sum(probabilities)
        l_ancestor = []
        for a in ancestors:
            for ii in range(nb):
                G.add_node(i)
                G.nodes[i]['quantizer'] = random.randint(1, 20) - 10
                G.nodes[i]['stage'] = t
                G.add_edge(a,i)
                G[a][i]['weight'] = probabilities[ii]
                l_ancestor.append(i)
                i = i+1
        ancestors = l_ancestor
    return(G)

def make_initial_tree(T):
    G = nx.DiGraph()
    i = 0
    ancestors = [0]
    G.add_node(0)
    G.nodes[i]['quantizer'] = random.randint(1, 20) - 10
    G.nodes[i]['stage'] = 0
    i = 1
    for t in range(1,T):
        nb = 2
        probabilities = np.random.random(nb)  # np.ones(len(children))
        probabilities /= np.sum(probabilities)
        l_ancestor = []
        for a in ancestors:
            for ii in range(nb):
                G.add_node(i)
                G.nodes[i]['quantizer'] = random.randint(1, 20) - 10
                G.nodes[i]['stage'] = t
                G.add_edge(a,i)
                G[a][i]['weight'] = probabilities[ii]
                l_ancestor.append(i)
                i = i+1
        ancestors = l_ancestor
    return(G)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pool_size = comm.Get_size()


    if False:
        l_methods = ['LP', 'MAM'] #, 'IBP']
        mean_time_LP = []
        mean_time_MAM = []
        mean_time_IBP = []
        gr = 600
        ec = 75
        arrangements = [[1,2,i] for i in range(1,gr+1) if i%ec==0 or i==1]
        for arrangement in arrangements:
            T = len(arrangement)
            H = make_growing_tree(T, arrangement)
            # draw_tree(H)
            G = make_initial_tree(T)
            print(f'Computing reduction for T={T}, and arrangement={arrangement} ...')
            RES = KP_reduction(H,G, method='LP', delta=1000,  itred=3)
            l = []
            for method in l_methods:
                l.append(np.mean([a[-1] for a in RES[method]['record_t_n']]))
            mean_time_LP.append(l[0])
            mean_time_MAM.append(l[1])
            # mean_time_IBP.append(l[2])
        name = f'outputs/mean_time_0.pkl'
        with open(name, 'wb') as f:
            pickle.dump([mean_time_LP, mean_time_MAM, mean_time_IBP], f)

        # plt.figure()
        # plt.plot([i for i in range(1,gr+1) if i%ec==0], mean_time_LP, label='LP')
        # plt.plot([i for i in range(1,gr+1) if i%ec==0], mean_time_MAM, label='MAM')
        # plt.plot([i for i in range(1,gr+1) if i%ec==0], mean_time_IBP, label='IBP')
        # plt.title('1 root and 2 nodes tree')
        # plt.xlabel('number of children at last stage')
        # plt.ylabel('computation time')
        # plt.grid()
        # plt.legend()
        # plt.show()


    if False:
        l_methods = ['LP', 'MAM'] #, 'IBP']
        mean_time_LP = []
        mean_time_MAM = []
        mean_time_IBP = []
        gr = 600
        ec = 75
        arrangements = [[1,2,100,i] for i in range(1,gr+1) if i%ec==0]
        for arrangement in arrangements:
            T = len(arrangement)
            H = make_growing_tree(T, arrangement)
            # draw_tree(H)
            G = make_initial_tree(T)
            print(f'Computing reduction for T={T}, and arrangement={arrangement} ...')
            RES = KP_reduction(H,G, method='LP', delta=1000,  itred=3)
            l = []
            for method in l_methods:
                l.append(np.mean([a[-1] for a in RES[method]['record_t_n']]))
            mean_time_LP.append(l[0])
            mean_time_MAM.append(l[1])
            # mean_time_IBP.append(l[2])

        name = f'outputs/mean_time_100.pkl'
        with open(name, 'wb') as f:
            pickle.dump([mean_time_LP, mean_time_MAM], f) #, mean_time_IBP

        plt.figure()
        plt.plot([i for i in range(1,gr+1) if i%ec==0], mean_time_LP, label='LP')
        plt.plot([i for i in range(1,gr+1) if i%ec==0], mean_time_MAM, label='MAM')
        # plt.plot([i for i in range(1,gr+1) if i%ec==0], mean_time_IBP, label='IBP')
        plt.title('1 root and 2 then 100 nodes tree')
        plt.xlabel('number of children at last stage')
        plt.ylabel('computation time')
        plt.grid()
        plt.legend()
        plt.show()

    if True:
        l_n = [i for i in range(1,11)] # if i%5==0]
        for n in l_n:
            l_methods = ['LP', 'MAM']  # , 'IBP']
            mean_time_LP = []
            mean_time_MAM = []
            mean_time_IBP = []
            gr = 600
            ec = 75
            arrangements = [[1, 2, n, i] for i in range(1, gr + 1) if i % ec == 0 or i==1]
            for arrangement in arrangements:
                T = len(arrangement)
                H = make_growing_tree(T, arrangement)
                draw_tree(H)
                G = make_initial_tree(T)
                print(f'Computing reduction for T={T}, and arrangement={arrangement} ...')
                RES = KP_reduction(H, G, method='LP', delta=1000, itred=3)
                l = []
                for method in l_methods:
                    l.append(np.mean([a[-1] for a in RES[method]['record_t_n']]))
                mean_time_LP.append(l[0])
                mean_time_MAM.append(l[1])
                # mean_time_IBP.append(l[2])

            name = f'outputs/mean_time_{n}_v2.pkl'
            with open(name, 'wb') as f:
                pickle.dump([mean_time_LP, mean_time_MAM], f)  # , mean_time_IBP
