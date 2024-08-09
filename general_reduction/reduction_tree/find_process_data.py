import numpy as np
import networkx as nx

# function to find the process, quantizers and probabilities of a process
def find_process_data(G):
    '''
    :param
    G: Scenario tree wher nodes contain 'quantizer' and 'stage' values. links (Tree[i][j]) contains 'weight' between linked nodes
    :return:
    path_G : list of list of path to the leaves (from 0 to i)
    p_weights_G: list of list of weights for the nodes of path_G
    p_proba_G: list of list of conditional probabilities between adjacent nodes of path_G
    max_stage_G: number of stages in the tree
    '''

    # get the leaves
    max_stage_G = np.max([G.nodes[i]['stage'] for i in G.nodes])
    leaves_G = [i for i in G.nodes if G.nodes[i]['stage'] == max_stage_G]

    # get the path
    path_G = []
    p_weights_G = []
    p_proba_G = []
    for i in leaves_G:
        path_i = np.array(nx.shortest_path(G, 0, i))
        path_G.append(path_i)
        p_weights_G.append(np.array([G.nodes[i]['quantizer'] for i in path_i]))
        p_proba_G.append(np.array([G[path_i[i]][path_i[i + 1]]['weight'] for i in range(len(path_i) - 1)]))

    # Output
    return(path_G, p_weights_G, p_proba_G, max_stage_G)