"""
This code aim is to propose an implementation of the computation of improving an approximate tree, following the method
proposed in 'Tree approximation for discrete time stochastic process: a process distance approach' from Kovacevic and Pichler.
"""



import matplotlib.pyplot as plt
import numpy as np
from visualization_tree import *
import networkx as nx
import random
import time
# my codes
from find_process_data import * # from two trees, get the matrix of process from each leaf
from LP_tree_reduction import *  # this code compute the LP needed at each stage of the original tree for each node of the approximated tree


def reduction_p_Kovacevic(H,G,Pi=np.zeros((2,2))):
    """
    :param H: Initial tree
    :param G: Approximated tree structure: only the filtration and the quantifiers are necessary
    H and G must have the same number of stages, the conditional probability of a node is stored in 'weight' and its stage in 'stage'
    :return:
    * G : an approximated tree of H obtained with the method described in 'Tree approximation for discrete time stochastic process:
    a process distance approach' from Kovacevic and Pichler
    * D_ij : the distance matrix between all nodes of H and G. Note that D_ij[0,0] is the nested distance betwen H and G.
    * Pi : the transport matrix between G and H
    * time_tot: total time in second to compute the reduction
    """

    # Time management
    start = time.time()

    # PARAMETERS
    # FIXME: this could be parameters in the function reduction_p_IBP_Sinkhorn
    # Careful: this parameter can cause a difference more or less significative between the exact distance_GH(H,G) and the
    # distance computed in this algorithm that prevent conditional probabilities to be null.
    # Note that this difference is also due to the approximation of the iterative Pi_hat from the initialization to the end
    epsilon = 10**-3


    # TRANSPORT MATRIX AND PROBA OF THE APPROXIMATED TREE
    # I build recursively the transport matrix thanks to the probabilities of the initial tree
    N1 = len(G.nodes)  # nb of nodes in G
    N2 = len(H.nodes)  # nb of nodes in H
    if np.sum(Pi) == 0:
        ## INITIALIZATION OF PI: this is needed because the recursive computation is based on the previous iteration,
        # therefore at iteration 0 it is based on the initialization
        Pi = np.zeros((N1, N2))
        Pi[0, 0] = 1
        # number of stages
        T = np.max([G.nodes[i]['stage'] for i in G.nodes])
        # Nodes of the stage T-1: so we start the recursivity with them
        ancestor_n = [i for i in G.nodes if G.nodes[i]['stage']==T-1]  # list_n = set(ancestor_n) follows
        ancestor_m = [i for i in H.nodes if H.nodes[i]['stage']==T-1]
        for t in range(T-1,-1,-1):
            # we go recursively from stage T-1 to stage 0, identifying each time the ancestors of the treated nodes
            list_m = set(ancestor_m)
            list_n = set(ancestor_n)
            ancestor_m = []
            ancestor_n = []
            for n in list_n:
                for m in list_m:
                    # Children of nodes n and m
                    children_n = [i for i in list(G[n]) if G.nodes[i]['stage'] == t+1]
                    children_m = [i for i in list(H[m]) if H.nodes[i]['stage'] == t+1]

                    # I uniformly fill the transport matrix using the constraint on the conditional proba of the original tree
                    for i in children_m:
                        for j in children_n:
                            Pi[j,i] = H[m][i]['weight'] / len(children_n)

                    # I collect the ancestor of nodes n and m
                    if t > 0:
                        ancestor_n.append([i for i in list(G[n]) if G.nodes[i]['stage'] == t - 1][0])
                        ancestor_m.append([i for i in list(H[m]) if H.nodes[i]['stage'] == t - 1][0])


    # INITAILIZATION OF THE DISTANCE MATRIX : exact process distance between the leaves
    # Distance matrix initialization
    D_ij = np.zeros((N1, N2))

    ai, wi, pi, T = find_process_data(H)
    aj, wj, pj = find_process_data(G)[:-1]
    # Using Euclidean distance
    leaves_H = len(ai)
    leaves_G = len(aj)
    for i in range(leaves_H):
        nodeH = ai[i][-1]
        for j in range(leaves_G):
            nodeG = aj[j][-1]
            D_ij[nodeG,nodeH] = np.sum((wi[i] - wj[j])**2)


    ## OPTIMIZATION OF THE PROBABILITIES
    # Nodes of the stage T-1: so we start the recursivity with them
    ancestor_m = [i for i in H.nodes if H.nodes[i]['stage']==T-1]
    ancestor_n = [i for i in G.nodes if G.nodes[i]['stage']==T-1]
    for t in range(T-1,-1,-1):
        # we go recursively from stage T-1 to stage 0, identifying each time the ancestors of the treated nodes
        list_m = set(ancestor_m)
        list_n = set(ancestor_n)
        ancestor_m = []
        ancestor_n = []
        for n in list_n:
            b = []  # list of probabilities
            c = []  # distance matrix with the ponderations
            children_n = [i for i in list(G[n]) if G.nodes[i]['stage'] == t+1]
            dist_matrices = []  # concatenation of distance matrices
            for m in list_m:
                # Children of nodes n and m
                children_m = [i for i in list(H[m]) if H.nodes[i]['stage'] == t+1]

                # Constraint on p
                p = np.array([H[m][i]['weight'] for i in children_m])
                b.append(p)

                # Distance matrix
                dist = D_ij[children_n, :]
                dist = dist[:, children_m]
                dist_matrices.append(dist)
                # ponderation on the distance matrix due to the initialization/previous iteration of Pi
                dist = dist * Pi[n,m]
                c.append(dist)

                # I collect the ancestor of node m for next step
                if t > 0:
                    ancestor_m.append([i for i in list(H[m]) if H.nodes[i]['stage'] == t - 1][0])

                # I collect the ancestor of node n for next step
            if t > 0:
                ancestor_n.append([i for i in list(G[n]) if G.nodes[i]['stage'] == t - 1][0])

            # Computation of the recursive distance at nodes m,n
            c = np.concatenate(c, axis=1)
            res = LP_reduction_nt(c, b)  # see LP_tree_reduction.py
            S = res[1]     # List of the number of children for each node m
            Pi_k = res[0]  # Conditional transport matrices, stacked with regard to m: (Pi_m1[j,i|n,m1] | Pi_m2 | ...)
            # print(res[2])

            # Fill the distance matrix and the transport matrix at node n, going through every node m of the stage t:
            index = 0
            for i_m,m in enumerate(list_m):
                # 3.3 of the article explains that Pi is not null: it is larger than a small number 'epsilon'
                # Pi_k[:,index:index+S[i_m]] = Pi_k[:,index:index+S[i_m]].clip(epsilon)
                Pi_k[:,index:index+S[i_m]] = Pi_k[:,index:index+S[i_m]] / np.sum(Pi_k[:,index:index+S[i_m]])  # The sum of all Pi[j,i|n,m] for n and m fixed is equal to 1

                # FILL the distance matrix
                D_ij[n, m] = np.sum( np.multiply(Pi_k[:,index:index+S[i_m]], dist_matrices[i_m]) )

                # FILL the transport matrix with the conditional probabilities
                # when the whole tree is treated, I will use (23) from 'Tree approximation for discrete time stochastic process:
                # a process distance approach' from Kovacevic and Pichler to build the Pi(i,j)
                children_m = [i for i in list(H[m]) if H.nodes[i]['stage'] == t + 1]

                # 3.3 of the article explains that Pi is not null: it is larger than a small number 'epsilon'
                Pi2 = Pi_k[:, index:index + S[i_m]]
                Pi2 = Pi2.clip(epsilon)
                Pi2 = Pi2 / np.sum(Pi2)
                for j_c, j in enumerate(children_n):
                    # Pi[j, children_m] = Pi_k[j_c, index:index + S[i_m]]
                    Pi[j, children_m] = Pi2[j_c,:]

                    # FILL the probabilities in the approximated tree
                    # I uniformly fill the transport matrix using the constraint on the conditional proba of the original tree
                    # I don't need to go through every nodes of a stage of the original tree, only one is sufficient:
                    if i_m == 0:
                        G[n][j]['weight'] = np.sum(Pi[j, children_m])
                        G[n][j]['weight'] = np.round(G[n][j]['weight'], 3)

                index = index + S[i_m]


    # REBUILD the updated Transport matrix between trees H and G
    Pi[0,0] = 1
    # Exact method:
    list_i1 = [i for i in H.nodes if H.nodes[i]['stage']==1]
    list_j1 = [i for i in G.nodes if G.nodes[i]['stage']==1]
    for t in range(1,T+1):
        for i1 in list_i1:
            ancestor_m = [i for i in list(H[i1]) if H.nodes[i]['stage'] == t - 1]
            for j1 in list_j1:
                ancestor_n = [i for i in list(G[j1]) if G.nodes[i]['stage'] == t - 1]
                for m in ancestor_m:
                    for n in ancestor_n:
                        Pi[j1,i1] = Pi[j1,i1] * Pi[n,m]
        list_i1 = [i for i in H.nodes if H.nodes[i]['stage']== t + 1]
        list_j1 = [i for i in G.nodes if G.nodes[i]['stage']== t + 1]


    # Time management:
    time_tot = time.time() - start
    return(G, D_ij, Pi, time_tot)










