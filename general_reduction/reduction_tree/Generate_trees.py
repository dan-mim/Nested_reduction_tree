
import matplotlib.pyplot as plt
import numpy as np
from visualization_tree import *
import networkx as nx
import random
from find_process_data import *
import time


# TREE GENERATION
# Original tree
def generate_tree(branchs_per_node, stages, rd1=42, rd2=42):
    """
    This function produces a tree with random quantizers (between 1 and 20) and probabilities of each branch
    :param branchs_per_node: number of branchs per node but more precisly number of children of each node
    :param stages: number of stages of the tree
    rd1 and rd2 are random states to generate the quantizers values and probabilities
    :return: H, the built tree
    """

    # seed for the quantizers of the two trees
    random.seed(rd1)
    # seed for the probabilities of the original tree
    np.random.seed(rd2)

    # Filtration
    nb_nodes = 0
    for i in range(stages):
        nb_nodes += branchs_per_node**i
    # generate the filtration
    H = nx.full_rary_tree(branchs_per_node, nb_nodes, create_using=None)

    # Assignation of quantizers for each nodes
    for i in range(len(H.nodes)):
        H.nodes[i]['quantizer'] = random.randint(1, 20) - 10  #int(random.random()*1000) #

    # Assignation of probabilities and stages
    ancestor = None
    H.nodes[0]['stage'] = 0
    edges = H.edges
    for e in edges:
        if e[0] != ancestor:
            ancestor = e[0]
            children = [edge[1] for edge in edges if edge[0]==ancestor]
            probabilities = np.random.random(len(children))   #np.ones(len(children))
            probabilities /= np.sum(probabilities)
            for i,child in enumerate(children):
                H[ancestor][child]['weight'] = probabilities[i]
                H.nodes[child]['stage'] = H.nodes[ancestor]['stage'] + 1

    # Output
    return(H)

# # Visualization
# draw_tree(G)
# draw_tree(H)
