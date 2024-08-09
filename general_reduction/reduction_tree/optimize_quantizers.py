"""
@daniel.mimouni
Implementation of the gradiant descent to improve the quantizers of trees in case of the quadratic Wasserstein
(Euclidean distance and Wasserstein of order r = 2).
"""

import numpy as np

def optim_quantizers(H, G, Pi):
    """
    This function compute the analytic optimization of the quantizers derived in Theorem 2 (25) in the case of
    Euclidean distance and Wasserstein of order 2 from 'Tree approximation for discrete time stochastic process:
    a process distance approach' from Kovacevic and Pichler
    :param H: Initial tree
    :param G: Approximated tree structure: only the filtration and the quantifiers are necessary
    H and G must have the same number of stages
    :param Pi: Transport matrix between H and G, this can be derived using one of the 3 algorithms 'tree_reduction---'
    :return:
    G the approximated tree updated with better quantizers
    """
    # number of stages
    T = np.max([G.nodes[i]['stage'] for i in G.nodes])

    ancestor_n = [i for i in G.nodes if G.nodes[i]['stage']==T]
    ancestor_m = [i for i in H.nodes if H.nodes[i]['stage']==T]
    for t in range(T,-1,-1):
        # we go recursively from stage T-1 to stage 0, identifying each time the ancestors of the treated nodes
        list_m = set(ancestor_m)
        list_n = set(ancestor_n)
        ancestor_m = []
        ancestor_n = []
        for n in list_n:
            den = 0
            num = 0
            for m in list_m:
                a = H.nodes[m]['quantizer']
                b = Pi[n,m]
                den = den + Pi[n,m] * H.nodes[m]['quantizer']
                num = num + Pi[n,m]

                # I collect the ancestor of node m for next step
                if t > 0:
                    ancestor_m.append( [i for i in H.predecessors(m)][0] )

            # FILL quantizer
            G.nodes[n]['quantizer'] = den / num

            # ancestors of node n
            if t > 0:
                ancestor_n.append([j for j in G.predecessors(n)][0] )
    # Output
    return(G)