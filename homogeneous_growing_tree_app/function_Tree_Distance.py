from find_process_data import *

# RECURSIVE COMPUTATION : SOLVE THE LINEAR PROGRAM
from scipy.sparse import csc_matrix
import scipy
from scipy import optimize
import time
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
    for i in range(leaves_G):
        nodeG = ai[i][-1]
        for j in range(leaves_H):
            nodeH = aj[j][-1]
            D_ij[nodeG,nodeH] = np.sum((wi[i] - wj[j])**2)


    ## I. METHOD THAT KEEPS THE TREE STRUCTURE
    # Nodes of the stage T-1: so we start the recursivity with them
    list_m = [i for i in G.nodes if G.nodes[i]['stage']==T-1]
    list_n = [i for i in H.nodes if H.nodes[i]['stage']==T-1]
    for t in range(T-1,-1,-1):
        # after stage T-1, I only use the ancestor of the used nodes
        if t < T-1:
            list_m = set(ancestor_m)
            list_n = set(ancestor_n)
        ancestor_m = []
        ancestor_n = []
        for m in list_m:
            for n in list_n:
                # Children of nodes n and m
                children_m = [i for i in list(G[m]) if G.nodes[i]['stage'] == t+1]
                children_n = [i for i in list(H[n]) if H.nodes[i]['stage'] == t+1]

                # Constraint on p and q
                p = np.array([G[m][i]['weight'] for i in children_m])
                q = np.array([H[n][i]['weight'] for i in children_n])

                # Distance matrix
                c = D_ij[children_m, :]
                c = c[:, children_n]

                # Computation of the recursive distance at nodes m,n
                p = np.abs(p / np.sum(p))
                q = np.abs(q / np.sum(q))
                c_max = np.max(c)
                res = LP_dlr_mn(c/c_max, p, q)
                if type(res[0]) != float:
                    res = LP_dlr_mn(np.round(c/c_max,3), p, q)
                # Recursivity is saved in D_ij
                D_ij[m, n] = res[0] * c_max
                # Pi = res[1]

                # D_ij[m,n] = np.sum( np.multiply(c,Pi) )

                # I collect the ancestor of nodes n and m
                if t > 0:
                    ancestor_m.append([i for i in list(G[m]) if G.nodes[i]['stage'] == t-1][0])
                    ancestor_n.append([i for i in list(H[n]) if H.nodes[i]['stage'] == t-1][0])
    # Output
    Time = time.time()-start
    return(D_ij[0,0], D_ij, Time)
