import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import pandas as pd

# my codes
from Generate_trees import *               # Generates tree
from function_Tree_Distance import *       # Compute the exact nested distance between two trees
from Tree_reduction import *               # Compute the optimal probabilities between 2 trees using the exact method of Kovacevic
from tree_reduction_IBP_sinkhorn import *  # Compute the optimal probabilities between 2 trees using IBP and Sinkhorn methods
from tree_reduction_MAM import *           # Compute the optimal probabilities between 2 trees using MAM
from optimize_quantizers import *          # Compute the optimal quantizers between 2 trees using the exact method of Kovacevic in the case of order 2

doc = 'données papier au 25 janv' #  'données papier au 22 fevr' #  "données papier au 25 janv" # 'données papier au 12 fevr rand 2020' # 'données papier au 9 fevr rand 1010'
rnd = None #21
random_or_not = f'_rnd{rnd}'
if rnd == None:
    random_or_not = ''
if 1==1:
    fig, axs = plt.subplots(2,3, figsize=(25,20))
    i = 0
    for ax in axs.ravel():
        cpnG = 2  # children per node approximate tree
        l_arr = [(4, 6), (5, 6), (6, 6), (7, 5), (7, 6), (8, 5)]
        T, cpnH = l_arr[i]
        i = i + 1
        print(f'{T} stages, initial tree has {cpnH} children per node and the approximate tree {cpnG}')
        # generate trees
        # H = generate_tree(cpnH, T, rd1=10, rd2=10)    # ORIGINAL TREE
        # G = generate_tree(cpnG, T, rd1=42, rd2=45)    # APPROXIMATE TREE


        # Plots
        for method in ['MAM', 'MAM4', 'IBP', 'Kovacevic']: #, 'MAM14' for T=7
            try:
                with open(f'{doc}/full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}{random_or_not}.pkl', 'rb') as f: #_rnd{rnd}
                    res = pickle.load(f)
                if (T, cpnH) == (5, 6):
                    with open(f'données papier au 12 fevr rand 2020/full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}_rnd{21}.pkl', 'rb') as f:  # _rnd{rnd}
                        res = pickle.load(f)
                l_dist0 = [a**.5 for a in res[-1][:]] # need to root the squared distance

                #Take into account the stopping critera of the reduction algorithm
                reduc_criterion = 0.1 #2
                v0 = l_dist0[0]
                l_dist = [v0]
                k = 1
                while np.abs(l_dist0[k]-v0) > reduc_criterion and k < len(l_dist0) - 2:
                    print(np.abs(l_dist0[k]-v0),k)
                    v0 = l_dist0[k]
                    l_dist.append(v0)
                    k = k + 1
                l_dist.append(l_dist0[k])
                print(l_dist)

                # styling the curves
                style = 'o-'
                label = method
                if method == 'MAM':
                    color = 'tab:blue'
                if method == 'MAM4':
                    style = 'o--'
                    label = 'MAM 4 processors'
                    color = 'tab:blue'
                if method == 'IBP':
                    color = 'tab:green'
                if method == 'Kovacevic':
                    label = 'LP'
                    color = 'tab:orange'
                temps_tot = np.round(np.sum(res[1][:len(l_dist)]), 2)
                if (T,cpnH) == (8,5):
                    temps_tot = np.round(np.sum(res[1][:len(l_dist)])/3, 2)
                # label = label + f' {temps_tot}s'
                x_label = np.linspace(1,len(l_dist),len(l_dist)) - 1
                ax.plot(x_label,l_dist,style, color=color,  label=label)
                ax.set_ylim(np.min(l_dist[1:])-1, np.max(l_dist[1:])+1)
            except:
                pass
        nb_of_nodes = cpnH ** (T - 1)
        for t in range(T - 2, 0, -1):
            nb_of_nodes += cpnH ** t
        nb_of_nodes += 1
        ax.set_title(f'{cpnH**(T-1)} scenarios, {nb_of_nodes} nodes, T = {T}, cpn = {cpnH}')
        ax.set_xlabel('iteration')
        ax.set_ylabel('Nested distance')
        ax.grid()
        ax.legend()
        # ax.set_xlim(0.9, 5.1)
        ax.axis([np.min(x_label), 6, y_min, y_max])
    plt.show()


if 1==1:
    fig, axs = plt.subplots(2,3, figsize=(25,20))
    i = 0
    for ax in axs.ravel():
        cpnG = 2  # children per node approximate tree
        l_arr = [(4, 6), (5, 6), (6, 6), (7, 5), (7, 6), (8, 5)]
        T, cpnH = l_arr[i]
        i = i + 1
        print(f'{T} stages, initial tree has {cpnH} children per node and the approximate tree {cpnG}')
        # generate trees
        # H = generate_tree(cpnH, T, rd1=10, rd2=10)    # ORIGINAL TREE
        # G = generate_tree(cpnG, T, rd1=42, rd2=45)    # APPROXIMATE TREE

        # Plots
        for method in ['MAM', 'MAM4', 'IBP', 'Kovacevic']: #, 'MAM14' for T=7
            try:
                with open(f'{doc}/full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}_rnd{rnd}.pkl', 'rb') as f: #_rnd{rnd}
                    res = pickle.load(f)
                l_tps = res[1]
                if (T,cpnH) == (8,5): #only case wher I used jupyterlab and not my own computer, and JL is at 2 to 3 times slower
                    l_tps = [a/2.5 for a in l_tps]
                style = 'o-'
                label = method
                if method == 'MAM':
                    color = 'tab:blue'
                if method == 'MAM10':
                    style='o--'
                    label = 'MAM 10 processors'
                    color = 'tab:blue'
                if method == 'IBP':
                    color = 'tab:green'
                if method == 'Kovacevic':
                    label = 'LP'
                    color = 'tab:orange'
                ax.plot(np.linspace(1, len(l_tps), len(l_tps)), l_tps, style, color=color, label=label)
            except:
                pass
        nb_of_nodes = cpnH**(T-1)
        for t in range(T-2,0,-1):
            nb_of_nodes += cpnH**t
        nb_of_nodes += 1
        ax.set_title(f'{cpnH**(T-1)} scenarios, {nb_of_nodes} nodes, T = {T}, cpn = {cpnH}')
        ax.set_xlabel('iteration')
        ax.set_ylabel('time per iteration (s)')
        ax.grid()
        ax.legend()
    plt.show()



# Other resuls
if 1==2:
    fig, axs = plt.subplots(1, 2, figsize=(18, 11))
    i = 0
    for ax in axs.ravel():
        T = [7,8][i]
        i = i + 1
    # for T in [4, 5, 6, 7]: # T is the number of stages
        cpnG = 2  # children per node approximate tree
        cpnH = 5  # children per node initial tree
        print(f'{T} stages, initial tree has {cpnH} children per node and the approximate tree {cpnG}')
        # generate trees
        # H = generate_tree(cpnH, T, rd1=10, rd2=10)    # ORIGINAL TREE
        # G = generate_tree(cpnG, T, rd1=42, rd2=45)    # APPROXIMATE TREE

        # Plots
        for method in ['MAM', 'IBP', 'Kovacevic']: #, 'MAM14' for T=7
            with open(f'full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}.pkl', 'rb') as f:
                res = pickle.load(f)
            l_tps = res[1]
            ax.plot(np.linspace(1, len(l_tps), len(l_tps)), l_tps, 'o-', label=method)
        ax.set_title(f'full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}')
        ax.set_xlabel('iteration')
        ax.set_ylabel('time per iteration (s)')
        ax.grid()
        ax.legend()
    plt.show()

if 1==2:
    fig, axs = plt.subplots(1,2, figsize=(18, 11))
    i = 0
    for ax in axs.ravel():
        T = [7,8][i]
        i = i + 1
    # for T in [4, 5, 6, 7]: # T is the number of stages
        cpnG = 2  # children per node approximate tree
        cpnH = 5  # children per node initial tree
        print(f'{T} stages, initial tree has {cpnH} children per node and the approximate tree {cpnG}')
        # generate trees
        # H = generate_tree(cpnH, T, rd1=10, rd2=10)    # ORIGINAL TREE
        # G = generate_tree(cpnG, T, rd1=42, rd2=45)    # APPROXIMATE TREE


        # Plots
        for method in ['MAM', 'IBP', 'Kovacevic']: #, 'MAM14' for T=7
            with open(f'full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}.pkl', 'rb') as f:
                res = pickle.load(f)
            l_dist = [a**.5 for a in res[-1][1:]] # need to root the squared distance
            ax.plot(np.linspace(1,len(l_dist),len(l_dist)),l_dist,'o-',  label=method)
        ax.set_title(f'full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}')
        ax.set_xlabel('iteration')
        ax.set_ylabel('Nested distance')
        ax.grid()
        ax.legend()
    plt.show()

# Get results in table
if 1==2:
    for T in [4, 5, 6, 7]:
    # for T in [4, 5, 6, 7]: # T is the number of stages
        cpnG = 2  # children per node approximate tree
        cpnH = 6  # children per node initial tree
        print(f'{T} stages, initial tree has {cpnH} children per node and the approximate tree {cpnG}')
        # generate trees
        # H = generate_tree(cpnH, T, rd1=10, rd2=10)    # ORIGINAL TREE
        # G = generate_tree(cpnG, T, rd1=42, rd2=45)    # APPROXIMATE TREE


        # DF
        df = pd.DataFrame({})
        for method in ['MAM', 'MAM10', 'IBP', 'Kovacevic']: #, 'MAM14' for T=7
            with open(f'full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}.pkl', 'rb') as f:
                res = pickle.load(f)
            l_tps = res[1]
            df[f'{method}'] = l_tps
        df.to_excel(f'full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}.xlsx')