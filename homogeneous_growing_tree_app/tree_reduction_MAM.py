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
from MAM_balanced import *  # # my implementation of MAM, that compûtes the barycenter between probability densities
from LP_Wasserstein_distance import *  # to compute the exact berycentric distance

def reduction_p_MAM(H,G,Pi=np.zeros((2,2))):
    """
    :param H: Initial tree
    :param G: Approximated tree structure: only the filtration and the quantifiers are necessary
    H and G must have the same number of stages
    :return:
    * G : an approximated tree of H obtained from a rewritting of 'Tree approximation for discrete time stochastic process:
    a process distance approach' from Kovacevic and Pichler taking advantage of the barycenter LP problem that enables to
    use the Method of Averaged Marginals (MAM) to solve the Linear Program
    * D_ij : the distance matrix between all nodes of H and G. Note that D_ij[0,0] is the nested distance betwen H and G.
    * Pi : the transport matrix between G and H
    * time_tot: total time in second to compute the reduction
    """

    # Time management
    start = time.time()

    # PARAMETERS
    # FIXME: this could be parameters in the function reduction_p_MAM
    rho = 5  # parameter for the computation of the MAM barycenter
    # Careful: this parameter can cause a difference more or less significative between the exact distance_GH(H,G) and the
    # distance computed in this algorithm that prevent conditional probabilities to be null.
    # Note that this difference is also due to the approximation of the iterative Pi_hat from the initialization to the end
    epsilon =  10 ** -3

    ## INITIALIZATION OF PI: this is needed because the recursive computation is based on the previous iteration,
    # therefore at iteration 0 it is based on the initialization
    # I build recursively the transport matrix thanks to the probabilities of the initial tree
    N1 = len(G.nodes)  # nb of nodes in G
    N2 = len(H.nodes)  # nb of nodes in H

    if np.sum(Pi) == 0:
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


    ## INITAILIZATION OF THE DISTANCE MATRIX : exact process distance between the leaves
    # Distance matrix initialization
    D_ij = np.zeros((N1, N2))

    ai, wi, pi, T = find_process_data(G)
    aj, wj, pj = find_process_data(H)[:-1]
    # Using Euclidean distance
    leaves_G = len(ai)
    leaves_H = len(aj)
    for i in range(leaves_G):
        nodeG = ai[i][-1]
        for j in range(leaves_H):
            nodeH = aj[j][-1]
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
            dist_matrices = []

            # go through each node and treat the subtree (node + its children)
            for m in list_m:
                # Children of nodes m
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

            # I reshape the probabilities vector as [p1,p2,p(m),0,0,0,...,0], [0,0,0,p1,p2,p(m),0,0,0,...,0] to keep one distance matrix as built in c
            sumS = np.array(b).size
            acc = 0
            b_ = []
            for p in b:
                v = np.zeros(sumS)
                v[acc:acc + len(p)] = p
                b_.append(v)
                acc += len(p)

            # I treat the resolution of the LP as a barycenter problem, using MAM:
            # this provides the barycenter AND the transport matrices !
            res = Operator_splitting_parallel(b_, c, computation_time=10, iterations_min=20, iterations_max=50, rho=5) #rho
            # Recursivity is saved in D_ij
            Pi_k = res[1]  # Conditional transport matrices, stacked with regard to m: Pi_k[m0], Pi_k[m1],...
            Pi_debug = Pi_k[0]
            # bary = res[0] / np.sum(res[0])
            # print(Wasserstein_distance_LP(bary, b, c)[0])

            # Fill the distance matrix and the transport matrix at node n, going through every nodes m of the stage t:

            # Parallelization initialization:
            # comm = MPI.COMM_WORLD
            # rank = comm.Get_rank()
            # pool_size = comm.Get_size()
            # splitting_work = division_tasks(len(list_m), pool_size)
            # for i_m in splitting_work[rank]:
            #     m = list_m[i_m]

            for i_m, m in enumerate(list_m):
                # 3.3 of the article explains that Pi is not null: it is larger than a small number 'epsilon'
                # Pi_k[i_m] = Pi_k[i_m].clip(epsilon)
                Pi_k[i_m] = Pi_k[i_m] / np.sum(Pi_k[i_m])  # The sum of all Pi[j,i|n,m] for n and m fixed is equal to 1

                # FILL the distance matrix
                D_ij[n, m] = np.sum(np.multiply(Pi_k[i_m], dist_matrices[i_m]))

                # FILL the transport matrix with the conditional probabilities
                # when the whole tree is treated, I will use (23) from 'Tree approximation for discrete time stochastic process:
                # a process distance approach' from Kovacevic and Pichler to build the Pi(i,j)
                children_m = [i for i in list(H[m]) if H.nodes[i]['stage'] == t + 1]
                sum_Gj = 0
                for j_c, j in enumerate(children_n):
                    Pi[j, children_m] = Pi_k[i_m][j_c, :]

                    # FILL the probabilities in the approximated tree
                    # I uniformly fill the transport matrix using the constraint on the conditional proba of the original tree
                    # I don't need to go through every nodes of a stage of the original tree, only one is sufficient:
                    if i_m == 0:
                        G[n][j]['weight'] = np.sum(Pi[j, children_m])
                        sum_Gj = sum_Gj + np.abs(G[n][j]['weight'])
                        G[n][j]['weight'] = np.round(G[n][j]['weight'], 3)
                # I force the branches from a node to be equal to 1 while HAVING POSITIVE VALUES!
                if i_m == 0:
                    for j_c, j in enumerate(children_n):
                        G[n][j]['weight'] = np.abs(G[n][j]['weight'] / sum_Gj )
                        G[n][j]['weight'] = np.round(G[n][j]['weight'], 3)

    # REBUILD the updated Transport matrix between trees H and G
    # Exact method:
    Pi[0,0] = 1
    # # method 1:
    # list_i1 = [i for i in H.nodes if H.nodes[i]['stage']==1]
    # list_j1 = [i for i in G.nodes if G.nodes[i]['stage']==1]
    # for t in range(1,T+1):
    #     for i1 in list_i1:
    #         ancestor_m = [i for i in list(H[i1]) if H.nodes[i]['stage'] == t - 1]
    #         for j1 in list_j1:
    #             ancestor_n = [i for i in list(G[j1]) if G.nodes[i]['stage'] == t - 1]
    #             for m in ancestor_m:
    #                 for n in ancestor_n:
    #                     Pi[j1,i1] = Pi[j1,i1] * Pi[n,m]
    #     list_i1 = [i for i in H.nodes if H.nodes[i]['stage']== t + 1]
    #     list_j1 = [i for i in G.nodes if G.nodes[i]['stage']== t + 1]

    # Method 2:
    # 3.3 of the article explains that Pi is not null: it is larger than a small number 'epsilon'
    # Pi = Pi.clip(epsilon)
    # Pi = Pi / np.sum(Pi)
    for t in range(T):
        children_n = [i for i in H.nodes if H.nodes[i]['stage']== t+1]
        children_m = [i for i in G.nodes if G.nodes[i]['stage']== t+1]
        Pi2 = Pi[:, children_n]
        Pi2 = Pi2[children_m,:]
        Pi2 = Pi2.clip(epsilon)
        Pi2 = Pi2 / np.sum(Pi2)
        for i_n,n in enumerate(children_n):
            Pi[children_m,n] = Pi2[:,i_n]

    # Time management:
    time_tot = time.time() - start
    return(G, D_ij, Pi, time_tot)










