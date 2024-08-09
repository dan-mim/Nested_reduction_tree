from reduction_tree.find_process_data import *

# RECURSIVE COMPUTATION : SOLVE THE LINEAR PROGRAM
from scipy.sparse import csc_matrix
import scipy
from scipy import optimize
import time
from mpi4py import MPI


def LP_dlr_mn(c,p,q):
    """
    This function is a linear program using the method High to compute (17) from the paper
    'Tree approximation for discrete time stochastic process: a process distance approach' from Kovacevic and Pichler
    Its aim is to recursively compute the distance between two trees.
    :param c: Distance matrix between i_s and j_s (R x S)
    :param p: constraint 1 P(i|m)
    :param q: constraint 2 P'(j|n)
    :return:
    Distance_exact = dlr(m,n),
    Pi_exact = pi(.,. | m,n)
    Time = time of execution
    """
    # trivial case
    # if len(p)==len(q) and np.sum([np.abs(p[i]-q[i]) for i in range(len(p))])==0:
    #     return(0,0)
    # Time management:
    start = time.time()

    # Objective function
    R, S = c.shape
    c = np.reshape(c, (R * S,))

    # Building A: Constraint (right sum of Pi to get p)
    shape_A = (R, R * S)
    indices_values_A = np.ones(R * S)
    # lines indices:
    line_indices_A = np.array([])
    for r in range(R):
        line_indices_A = np.append(line_indices_A, np.ones(S) * r)
    # column indices:
    column_indices_A = np.linspace(0, R * S - 1, R * S)

    # Building B: Constraint (left sum of Pi to get q)
    shape_B = (S, R*S)
    indices_values_B = np.ones(R*S)
    # lines indices:
    line_indices_B = np.array([])
    for s in range(S):
        line_indices_B = np.append(line_indices_B, np.ones(R) * s)
    # column indices: time_spent for R_=S_=1600
    column_indices_B = np.array([])
    for s in range(S):
        column_indices_B = np.append(column_indices_B, np.linspace(0,S*(R-1),R) + s)

    # Building A_eq = concatenate((A, B), axis=0)
    shape_Aeq = (R + S, R * S)
    indices_values_Aeq = np.append(indices_values_A, indices_values_B)
    line_indices_Aeq = np.append(line_indices_A, line_indices_B + R)
    column_indices_Aeq = np.append(column_indices_A, column_indices_B)
    A_eq = csc_matrix((indices_values_Aeq, (line_indices_Aeq, column_indices_Aeq)), shape_Aeq)

    # b_eq:
    b_eq = np.append(p, q)

    # Resolution: with the bound x >= 0
    AEQ =  csc_matrix.todense(A_eq)
    res = scipy.optimize.linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=(0,None), method='highs')  #, bounds=(0,None)

    # computing time:
    end = time.time()
    Time = np.round((end - start), 2)

    # Output:
    # Optimization terminated successfully.
    Distance_exact = res.fun
    # Pi_exact = res.x
    # Pi_exact = np.reshape(Pi_exact, (R, S))

    return(Distance_exact, Time)  #Pi_exact

# COMPUTE DISTANCE BETWEEN TWO TREES:
def distance_GH(G,H):
    """
    This function compute the distance between the trees G and H, as derived in (17) from
    'Tree approximation for discrete time stochastic process: a process distance approach' from Kovacevic and Pichler
    :param G: Tree structure built in networkx
    :param H: Tree structure built in networkx
    :return: D_ij[0,0], D_ij, Time : The nested distance between the two trees is {D_ij[0,0]}, computerd in {Time} s
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pool_size = comm.Get_size()

    # Manage Time
    start = time.time()

    # DISTANCE MATRIX BETWEEN NODES
    N1 = len(G.nodes)
    N2 = len(H.nodes)
    D_ij = np.zeros((N1, N2))

    # DISTANCE BETWEEN THE LEAVES
    # I need to find all the scenarios from the root to the leaves to compute the distance between leaves
    ai, wi, pi, T = find_process_data(G)
    aj, wj, pj = find_process_data(H)[:-1]
    # Using Euclidean distance
    leaves_G = len(ai)
    leaves_H = len(aj)
    nodesG = []
    nodesH = []
    D_ij = np.zeros((leaves_G, leaves_H))
    for i in range(leaves_G):
        nodesG.append(ai[i][-1])
        for j in range(leaves_H):
            D_ij[i, j] = np.sum((wi[i] - wj[j]) ** 2)
            if i == 0:
                nodesH.append(aj[j][-1])


    ## I. METHOD THAT KEEPS THE TREE STRUCTURE
    # Nodes of the stage T-1: so we start the recursivity with them
    list_m = [i for i in G.nodes if G.nodes[i]['stage']==T-1]
    list_n = [i for i in H.nodes if H.nodes[i]['stage']==T-1]
    for t in range(T-1,-1,-1):
        # after stage T-1, I only use the ancestor of the used nodes
        if t < T-1:
            list_m = set(ancestor_m)
            list_n = set(ancestor_n)

        # Parrallel work using MPI:
        outputs = {}
        splitting_work = division_tasks(len(list_m), pool_size)
        for work in splitting_work[rank]:
            m = list(list_m)[work]
            outputs[m] = _loop_n(t, G, H, list_n, nodesH, nodesG, D_ij, m)

        l_outputs = comm.gather(outputs, root=0)
        if rank == 0:
            for dico in l_outputs:
                for key in dico.keys():
                    outputs[key] = dico[key]
        outputs = comm.bcast(outputs, root=0)

        D_ij = []
        ancestor_m =[]
        for i_m,m in enumerate(list_m):
            D_ij.append(outputs[m]["D_ij_t"])
            ancestor_m.extend(outputs[m]["ancestor_m"])
            if i_m == 0:
                ancestor_n = outputs[m]["ancestor_n"]
        D_ij = np.array(D_ij)
        nodesG, nodesH = list(list_m), list(list_n)

    # Output
    Time = time.time()-start
    return(D_ij[0,0], D_ij, Time)

def _loop_n(t, G, H, list_n, nodesH, nodesG, D_ij, m):
    ancestor_m = []
    ancestor_n = []
    D_ij_t = np.zeros(len(list_n))
    for i_n, n in enumerate(list_n):
        # Children of nodes n and m
        children_n = [i for i in H.successors(n)]
        children_ni = [j for j in range(len(nodesH)) if nodesH[j] in children_n]
        children_m = [i for i in G.successors(m)]
        children_mj = [i for i in range(len(nodesG)) if nodesG[i] in children_m]

        # Constraint on p and q
        p = np.array([G[m][i]['weight'] for i in children_m])
        q = np.array([H[n][i]['weight'] for i in children_n])

        # Distance matrix
        c = D_ij[children_mj, :]
        c = c[:, children_ni]

        # Computation of the recursive distance at nodes m,n
        p = np.abs(p / np.sum(p))
        q = np.abs(q / np.sum(q))
        c_max = np.max(c)
        res = LP_dlr_mn(c / c_max, p, q)
        # if type(res[0]) != float:
        #     res = LP_dlr_mn(np.round(c/c_max,3), p, q)
        # Recursivity is saved in D_ij
        D_ij_t[i_n] = res[0] * c_max

        # I collect the ancestor of nodes n and m
        if t > 0:
            ancestor_n.append([i for i in H.predecessors(n)][0])
            ancestor_m.append([i for i in G.predecessors(m)][0])

    # Outputs
    return dict(D_ij_t=D_ij_t, ancestor_n=ancestor_n, ancestor_m=ancestor_m)


def division_tasks(nb_tasks, pool_size):
    """
    Inputs: (int)
    *nb_tasks
    *pool_size : number of CPU/GPU to divide the tasks between

    Outputs:
    rearranged: numpy list of lists so that rearranged[i] should be treated by CPU[i] (rank=i)
    """
    # The tasks can be equaly divided for each CPUs
    if nb_tasks % pool_size == 0:
        rearranged = np.array([i for i in range(nb_tasks)])
        rearranged = np.split(rearranged, pool_size)

    # Some CPUs will receive more tasks
    else:
        div = nb_tasks // pool_size
        congru = nb_tasks % pool_size
        rearranged1 = np.array([i for i in range(div * congru + congru)])
        rearranged1 = np.split(rearranged1, congru)
        rearranged2 = np.array([i for i in range(div * congru + congru, nb_tasks)])
        rearranged2 = np.split(rearranged2, pool_size - congru)
        rearranged = rearranged1 + rearranged2

    # Output:
    return (rearranged)