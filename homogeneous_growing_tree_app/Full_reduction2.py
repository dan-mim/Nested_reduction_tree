"""
This code use probability optimization and quantizer optimization to compute the most improved approximation of a tree
"""
import matplotlib.pyplot as plt
import numpy as np
import pickle
from mpi4py import MPI

# my codes
from Generate_trees import *               # Generates tree
from function_Tree_Distance import *       # Compute the exact nested distance between two trees
from Tree_reduction import *               # Compute the optimal probabilities between 2 trees using the exact method of Kovacevic
from tree_reduction_IBP_sinkhorn import *  # Compute the optimal probabilities between 2 trees using IBP and Sinkhorn methods
from tree_reduction_MAM import *           # Compute the optimal probabilities between 2 trees using MAM
from optimize_quantizers import *          # Compute the optimal quantizers between 2 trees using the exact method of Kovacevic in the case of order 2

print('A FAIRE TOURNER SI ORDI BRANCHE ET PAS EN ECO D ENERGIE !!')

def full_reduction(H, G, method='Kovacevic', Pi=np.zeros((2,2))):
    """

    :param H: Initial tree that needs to be reduced
    :param G: Approximate tree
    :param method: (str) the method to compute the : can be -'Kovacevic' (use LP to compute the barycenter)[see Nested reductionfrom Kovacevic and Pichler],
                                                            -'MAM' [see the Method of averaged marginales from Mimouni et al.],
                                                            -'IBP' it will use IBP and Snkhorn [see Peyré's work],
    :param Pi: The optimal transport matrix between the trees G and H can be initialized with another method, if not it is set as zero.
    :return:
    G: the approximate tree
    l_tps: the computation time of the reduction method for the probabilities only
    l_dist:
    """

    # keep track
    start = time.time()
    l_tps = []
    l_G = []
    l_dist = []
    # # test debugage
    # G2 = G.copy()
    # a0 = distance_GH(G, G2)[0]
    # Pi2 = Pi.copy()
    # # fin test debugage
    l_iter = np.linspace(1,7, 7)
    for iter in l_iter[:-1]:
        # print(iter)

        # PROBABILITY OPTIMIZATION
        if method=='Kovacevic':
            res = reduction_p_Kovacevic(H, G) #, Pi)
        if method=='MAM':
            res = reduction_p_MAM(H,G) #,Pi)  #I used Operator_splitting_parallel(b_, c, computation_time=10, iterations_min=20, iterations_max=200, rho=5)
                                           # + internal criterion in MAM_balanced.py : evol_p>10**-6
            # # test debugage
            # a1 = distance_GH(G, G2)[0]
            # # if iter > 1:
             # res2 = reduction_p_Kovacevic(H,G2,Pi2)
            # a2 = distance_GH(G, G2)[0]
            # G2 = res2[0].copy()
            # Pi2 = res2[2].copy()
            # # fin test debug
        if method=='IBP':
            res = reduction_p_IBP_Sinkhorn(H,G) #,Pi)
        G = res[0].copy()
        Pi= res[2].copy()
        # # test debugage
        # Pi_diff = Pi - Pi2
        # pi_sum = np.sum(np.abs(Pi_diff))
        # a = distance_GH(G, G2)[0]
        # # draw_tree(G)
        # # draw_tree(G2)
        # G2 = optim_quantizers(H, G2, Pi2)
        # # fin test debug
        # draw_tree(G_MAM)

        l_tps.append(res[-1])
        l_G.append(G) ######################## I usually put it after the quantizer optim
        # print(f'{method} time ', res[-1])

        # QUANTIZER OPTIMIZATION
        G = optim_quantizers(H, G, Pi)


    # Output
    total_time = time.time() - start
    return(G, l_tps, l_G, total_time) #, l_dist)

rnd = 21
if 1 == 2:
    # Compute the reduction method and compute their nested distance each steps
    # parallel work for MAM multiprocessors
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pool_size = comm.Get_size()

    # for different trees:
    case = 0
    for arr in [(4,6), (5,6), (6,6)]: #(5,6), (4,6), (6,6), (7,5), (7,6), (8,5)]: #
        case = case + 1
        T, cpnH = arr[0], arr[1] # T is the number of stages and cpnH children per node initial tree
        cpnG = 2  # children per node approximate tree
        # cpnH = 5  # children per node initial tree
        print(f'{T} stages, initial tree has {cpnH} children per node and the approximate tree {cpnG}')

        # generate trees
        # if cpnH not in [5,7]:
        #     H = generate_tree(cpnH, T, rd1=10, rd2=10)      # ORIGINAL TREE  (seeds rd1=10, rd2=10)
        # if cpnH in [5,7]:
        H = generate_tree(cpnH, T, rd1=rnd, rd2=rnd)          # ORIGINAL TREE  (seeds rd1=20, rd2=20) autre tree tried: 21,21
        G = generate_tree(cpnG, T, rd1=42, rd2=45)          # APPROXIMATE TREE
        initial_distance = distance_GH(H, G)[0]

        # Compute the reduction for different methods:
        # Cases I need
        if case in [0,1,2,3,4,5]:
            l_methods = ['IBP', 'Kovacevic', 'MAM'] #, 'MAM', 'Kovacevic']
        # if case==4:
        #     l_methods = ['MAM', 'Kovacevic']
        # if case==5:
        #     l_methods = ['IBP', 'MAM', 'Kovacevic']
        if pool_size > 1: # for multiprocessing, only MAM is adequat
            l_methods = ['MAM']

        # Computation of the reductions
        for method in l_methods:
            print(f'computing with {method}...')
            res_reduc = full_reduction(H, G, method=method)

            # Save work datas
            txt_pool_size = pool_size
            if pool_size == 1 or method not in ['MAM']:
                txt_pool_size = ''
            # if rank == 0:
            #     with open(f'données papier au 9 fevr/full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}{pool_size}.pkl', 'wb') as f:
            #         pickle.dump(res_reduc, f)

        # Computation of the nested distances
            l_dist = [initial_distance]
            # Compute the nested distances after each iteration
            print(f'Compute the  nested distance for {method}')
            for G_ in res_reduc[2]:
                dist = distance_GH(H, G_)[0]
                l_dist.append(dist)
            res_reduc = res_reduc + (l_dist,)
            # store the new completed (with the nested distances) data:
            if rank == 0:
                with open(f'données papier au 22 fevr/full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}{txt_pool_size}_rnd{rnd}_method2.pkl', 'wb') as f:
                    pickle.dump(res_reduc, f)
            res_reduc = []  # emptying the RAM


# Comparison for multiple initialization
rnd = 21
if 1 == 1:
    # Compute the reduction method and compute their nested distance each steps
    # parallel work for MAM multiprocessors
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pool_size = comm.Get_size()

    # for different trees:
    case = 0
    arr = (5,6)
    for rnd in [22, 23, 24, 25]: #(5,6), (4,6), (6,6), (7,5), (7,6), (8,5)]: #
        T, cpnH = arr[0], arr[1] # T is the number of stages and cpnH children per node initial tree
        cpnG = 2  # children per node approximate tree
        # cpnH = 5  # children per node initial tree
        print(f'{T} stages, initial tree has {cpnH} children per node and the approximate tree {cpnG}, random state = {rnd}')

        # generate trees
        # if cpnH not in [5,7]:
        #     H = generate_tree(cpnH, T, rd1=10, rd2=10)      # ORIGINAL TREE  (seeds rd1=10, rd2=10)
        # if cpnH in [5,7]:
        H = generate_tree(cpnH, T, rd1=40, rd2=40)            # ORIGINAL TREE  (seeds rd1=20, rd2=20) autre tree tried: 21,21
        G = generate_tree(cpnG, T, rd1=rnd, rd2=rnd)          # APPROXIMATE TREE
        initial_distance = distance_GH(H, G)[0]

        # Compute the reduction for different methods:
        # Cases I need
        l_methods = ['IBP', 'Kovacevic', 'MAM']
        if pool_size > 1: # for multiprocessing, only MAM is adequat
            l_methods = ['MAM']

        # Computation of the reductions
        for method in l_methods:
            print(f'computing with {method}...')
            res_reduc = full_reduction(H, G, method=method)

            # Save work datas
            txt_pool_size = pool_size
            if pool_size == 1 or method not in ['MAM']:
                txt_pool_size = ''
            # if rank == 0:
            #     with open(f'données papier au 9 fevr/full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}{pool_size}.pkl', 'wb') as f:
            #         pickle.dump(res_reduc, f)

        # Computation of the nested distances
            l_dist = [initial_distance]
            # Compute the nested distances after each iteration
            print(f'Compute the  nested distance for {method}')
            for G_ in res_reduc[2]:
                dist = distance_GH(H, G_)[0]
                l_dist.append(dist)
            res_reduc = res_reduc + (l_dist,)
            # store the new completed (with the nested distances) data:
            if rank == 0:
                with open(f'comparison_multipl_rnd/full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}{txt_pool_size}_rnd{rnd}_method2.pkl', 'wb') as f:
                    pickle.dump(res_reduc, f)
            res_reduc = []  # emptying the RAM



# Compute nested distances and store them in the .pkl
if 1 == 2:
    for T in [5]:  # T is the number of stages
        cpnG = 2  # children per node approximate tree
        cpnH = 5  # children per node initial tree
        print(f'Compute nested distances for {T} stages, initial tree has {cpnH} children per node and the approximate tree {cpnG}')
        # generate initial trees
        if cpnH not in [5,7]:
            H = generate_tree(cpnH, T, rd1=10, rd2=10)      # ORIGINAL TREE  (seeds rd1=10, rd2=10)
        if cpnH in [5,7]:
            H = generate_tree(cpnH, T, rd1=21, rd2=21)     # ORIGINAL TREE
        G = generate_tree(cpnG, T, rd1=42, rd2=45)        # APPROXIMATE TREE
        initial_distance = distance_GH(H, G)[0]

        # Plots
        for method in ['IBP', 'MAM', 'Kovacevic']: #, 'Kovacevic']:  # 'MAM',  'IBP', '5000MAM',, 'MAM14' for T=7
            l_dist = [initial_distance]
            with open(f'full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}.pkl', 'rb') as f:
                res = pickle.load(f)
            # Compute the nested distances after each iteration
            print(f'Compute the distance for {method}')
            for G_ in res[2]:
                dist = distance_GH(H, G_)[0]
                l_dist.append(dist)
            res = res + (l_dist,)
            # store the new completed (with the nested distances) data:
            with open(f'full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}.pkl', 'wb') as f:
                pickle.dump(res, f)

if 1 ==2:
    # Compute the reduction method
    # for different trees:
    for T in [7]: # T is the number of stages #5,6,
        cpnG = 2  # children per node approximate tree
        cpnH = 6  # children per node initial tree
        print(f'{T} stages, initial tree has {cpnH} children per node and the approximate tree {cpnG}')
        # generate trees
        H = generate_tree(cpnH, T, rd1=10, rd2=10)    # ORIGINAL TREE  (seeds rd1=10, rd2=10)
        G = generate_tree(cpnG, T, rd1=42, rd2=45)    # APPROXIMATE TREE

        # Compute the reduction for different methods:
        for method in ['Kovacevic']:
            print(f'computing with {method}...')
            res_reduc = full_reduction(H, G, method=method)
            comm = MPI.COMM_WORLD
            rank = comm.Get_rank()
            pool_size = comm.Get_size()
            if pool_size==1 or method not in ['MAM']:
                pool_size=''
            if rank==0:
                with open(f'full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}{pool_size}.pkl','wb') as f:
                    pickle.dump(res_reduc, f)

# draw_tree(H)
# initial_distance = distance_GH(H,G)[0]
# print(f"Initial distance between the trees: {(initial_distance)**.5}")

# with open(f'full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}.pkl', 'rb') as f:
#     b = pickle.load(f)

# Compute nested distances and store them in the .pkl
if 1==2:
    for T in [7]: # T is the number of stages
        cpnG = 2  # children per node approximate tree
        cpnH = 6  # children per node initial tree
        print(f'Compute nested distances for {T} stages, initial tree has {cpnH} children per node and the approximate tree {cpnG}')
        # generate initial trees
        # H = generate_tree(cpnH, T, rd1=10, rd2=10)    # ORIGINAL TREE
        G = generate_tree(cpnG, T, rd1=42, rd2=45)    # APPROXIMATE TREE
        initial_distance = distance_GH(H, G)[0]

        # Plots
        for method in ['Kovacevic']: #'MAM',  'IBP', '5000MAM',, 'MAM14' for T=7
            l_dist = [initial_distance]
            with open(f'full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}.pkl', 'rb') as f:
                res = pickle.load(f)
            # Compute the nested distances after each iteration
            print(f'Compute the distance for {method}')
            for G_ in res[2]:
                dist = distance_GH(H, G_)[0]
                l_dist.append(dist)
            res = res + (l_dist,)
            # store the new completed (with the nested distances) data:
            with open(f'full_reduction_T{T}_cpnH{cpnH}_cpnG{cpnG}_{method}.pkl', 'wb') as f:
                pickle.dump(res, f)
                
