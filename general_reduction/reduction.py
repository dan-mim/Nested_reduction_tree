"""
@daniel.mimouni
This code aim at producing a reduced number of scenarios that approximate the better the initial large number
of produced scenarios using the Kovacevic and Pichler algorithm, improved with the MAM (or IBP) methods to gain speed
"""

# General
import sys
from mpi4py import MPI
# My codes
from reduction_tree.tree_reduction_MPI import *
from reduction_tree.optimize_quantizers import *
from reduction_tree.tree_distance_MPI import *
# from reduction_tree.function_Tree_Distance import * #function_Tree_Distance
from reduction_tree.visualization_tree import *

def KP_reduction(H,G, method='LP', delta=1000,  itred=7, npool=1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pool_size = comm.Get_size()

    # initial_dist = distance_GH(G, H)[0]
    # if rank == 0 :
    #     print(f"initial_dist={initial_dist}")
    #     sys.stdout.flush()

    RES = {}
    l_methods = ['MAM', 'LP'] #, 'IBP'] #, 'IBP'
    if pool_size >1:
        l_methods = ['MAM']
    for method in l_methods:
        st1 = time.time()
        if rank == 0:
            # print(f"method {method} is computing the tree reduction...")
            sys.stdout.flush()
        res = full_reduction(H, G, iterations=itred, method=method, npool=npool, rank=rank, delta=delta)
        end = time.time()
        RES[method] = res
        if rank == 0:
            # print(f"method {method} took {np.round(end-st1,2)}s and found an approached ND of {res['ND_aprx']}")
            sys.stdout.flush()
            T = G.nodes[np.max(G.nodes())]['stage'] + 1
            name = f'outputs/T{T}_mpi{pool_size}.pkl'
            with open(name, 'wb') as f:
                pickle.dump(RES,f)
        # distance = distance_GH(res['G'], H)[0]
        # if rank == 0:
        #     print(f"method {method} found a solution of ND = {distance}")
        #     sys.stdout.flush()

    return(RES)

def KP_reduced_tree(filtration1, aggregated_scens1, probas1, filtration2, aggregated_scens2, probas2, method='LP', delta=1000,  itred=7, npool=1):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pool_size = comm.Get_size()
    # if rank == 0:
    #     print(f"This reduction method uses the {method} method with the recursive KP algorithm and performs {itred} iterations")
    #     sys.stdout.flush()

    # Turn filtrations into trees using networkx defintions:
    H = into_networkx_tree(filtration1, aggregated_scens1, probas1)
    G = into_networkx_tree(filtration2, aggregated_scens2, probas2)
    # draw_tree(G)
    if rank ==0:
        print(f"strucutre du H={len(H.nodes)} noeuds, strucutre du G={len(G.nodes)} noeuds")

    # Reduce tree H into an approximated tree using the filtration of tree G
    # the probabilities and quantizers of the approximated tree are initialized with tree G as well
    # res = full_reduction(H, G, iterations=itred, method=method, rank=rank)
    # G_improved = res[0]

    initial_dist = distance_GH(G, H)[0]
    if rank == 0 :
        print(f"initial_dist={initial_dist}")
        sys.stdout.flush()

    l_methods = [ 'LP','MAM' ] #'IBP',
    if pool_size >1:
        l_methods = ['MAM']
    for method in l_methods:
        st1 = time.time()
        if rank == 0:
            print(f"method {method} is computing the tree reduction...")
            sys.stdout.flush()
        res = full_reduction(H, G, iterations=itred, method=method, npool=npool, rank=rank, delta=delta)
        if rank == 0:
            print(f"method {method} took {np.round(time.time()-st1,2)}s and found an approached ND of {res['ND_aprx']}")
            sys.stdout.flush()
        distance = distance_GH(res['G'], H)[0]
        if rank == 0:
            print(f"method {method} found a solution of ND = {distance}")
            sys.stdout.flush()

    return

    # # Get back the filtration, scenarios and probabilities of the improved tree G_improved
    # list_nodes, list_scenarios, probas_cond, depth = find_process_data(G_improved)
    # probas = [np.prod(q) for q in probas_cond]
    # # Another method can be to utilize the property of the tree with these functions: list_scenarios, probas  = retrieve_scenario_from_tree(H)
    # # (these are recursive functions)
    #
    # # Output
    # return(list_scenarios, probas)


def into_networkx_tree(filtration, aggregated_scens, probas): # (filtration: list[list[Atom]], aggregated_scens: list, probas: list):
    G = nx.DiGraph() # nx.Graph()
    stage = 0
    i = 0
    ancestor= 0
    for l_atoms in filtration:
        for atom in l_atoms:
            G.add_node(i)
            atom.set_node(i)
            if i > 0:
                ancestor_atom = atom.parent
                ancestor = ancestor_atom.node
                p_a = np.sum([probas[s] for s in ancestor_atom._atom])
                G.add_edge(ancestor,i)
                # p(b|a) = p(b,a)/p(a) = p(b)/p(a):
                G[ancestor][i]['weight'] = np.sum([probas[s] for s in atom._atom]) / p_a
            G.nodes[i]['quantizer'] = np.sum(np.array([ aggregated_scens[s][:,stage-1] for s in atom._atom]), axis=0)
            G.nodes[i]['stage'] = stage
            i += 1
        stage += 1

    return(G)


def full_reduction(H, G, method='LP', Pi=np.zeros((2,2)), iterations=7, keep_track=True,npool=1, rank=0, delta=1000):
    # """
    #
    # :param H: Initial tree that needs to be reduced
    # :param G: Approximate tree
    # :param method: (str) the method to compute the : can be -'Kovacevic' (use LP to compute the barycenter)[see Nested reductionfrom Kovacevic and Pichler],
    #                                                         -'MAM' [see the Method of averaged marginales from Mimouni et al.],
    #                                                         -'IBP' it will use IBP and Snkhorn [see Peyré's work],
    # :param Pi: The optimal transport matrix between the trees G and H can be initialized with another method, if not it is set as zero.
    # :return:
    # G: the approximate tree
    # l_tps: the computation time of the reduction method for the probabilities only
    # l_dist:
    # """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    pool_size = comm.Get_size()

    # keep track
    start = time.time()
    l_tps = []
    l_G = []
    Pi = 0
    ND_aprx_precedent = 0
    record_t_n = []
    for iter in range(iterations):
        if rank == 0:
            print(f"iteration {iter} of the reduction tree algorithm")
            sys.stdout.flush()
            st = time.time()

        # PROBABILITY OPTIMIZATION
        res = reduction_tree(H,G,Pi, method=method, npool=npool)

        G = res[0]
        Pi= res[2]
        ND_aprx = res[1]
        nb_n = res[4]
        nb_m = res[5]
        record_t_n.append(res[3])

        if keep_track:
            l_tps.append(res[-1])
            l_G.append(G)

        # QUANTIZER OPTIMIZATION
        G = optim_quantizers(H, G, Pi)

        if rank == 0:
            print(f"the reduction iteration {iter} took {np.round(time.time() - st, 2)}s, and found a (approximated if the method is not LP) ND of {ND_aprx}")

        # Stopping criterion:
        if rank ==0:
            if np.abs(ND_aprx - ND_aprx_precedent) < delta:
                print(f'The stopping criterion is reached !')
                sys.stdout.flush()

            # Save
            T = G.nodes[np.max(G.nodes())]['stage'] + 1
            name = f'outputs/T{T}_mpi{pool_size}.pkl'
            with open(name, 'wb') as f:
                pickle.dump(dict(G=G, l_tps=l_tps, l_G=l_G, record_t_n=record_t_n, nb_n=nb_n, nb_m=nb_m, ND_aprx=ND_aprx), f)
        ND_aprx_precedent = ND_aprx

    if rank == 0:
        print(f"the reduction ({iterations} iterations) took {np.round(time.time() - start, 2)}s")

    # Output
    return dict(G=G, l_tps=l_tps, l_G=l_G, record_t_n=record_t_n, nb_n=nb_n, nb_m=nb_m, ND_aprx=ND_aprx)


def retrieve_scenario_from_tree(G):
    # get the leaves
    max_stage_G = np.max([G.nodes[i]['stage'] for i in G.nodes])
    leaves_G = [i for i in G.nodes if G.nodes[i]['stage'] == max_stage_G]

    root = 0
    shape_scenario = (G.nodes[root]['quantizer'].shape[0], max_stage_G)
    scenario_numbers = len(leaves_G)
    list_scenarios = []
    explored = set()
    explored.add(root)
    this_scenario = []
    this_proba = []
    probas = []
    _dfs_filtration(root, G, explored, list_scenarios, probas, this_scenario, this_proba, shape_scenario, scenario_numbers)
    return list_scenarios, probas


def _dfs_filtration(nodenb, G, explored, list_scenarios, probas, this_scenario, this_proba, shape_scenario, scenario_numbers):
    t = G.nodes[nodenb]['stage']
    if len(this_scenario) > t:
        this_scenario = [list_scenarios[-1][:, i].reshape((shape_scenario[0], 1)) for i in range(t)]
        this_proba = [probas[-1][i] for i in range(t)]

    this_scenario.append(G.nodes[nodenb]['quantizer'].reshape((shape_scenario[0], 1)))
    if nodenb == 0:
        this_proba.append(1.)
    if nodenb > 0:
        parent = [i for i in list(G[nodenb]) if G.nodes[i]['stage'] == t - 1][0]
        this_proba.append(G[nodenb][parent]['weight'])

    children = [i for i in list(G[nodenb]) if G.nodes[i]['stage'] == t + 1]
    if len(children) > 0:
        for child_atom in children:
            if child_atom not in explored:
                explored.add(child_atom)
                _dfs_filtration(child_atom, G, explored, list_scenarios, probas, this_scenario, this_proba, shape_scenario, scenario_numbers)
    else:
        probas.append(this_proba)
        list_scenarios.append(np.concatenate(this_scenario, axis=1))
