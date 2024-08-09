"""
In this version of the Sinkhorn algorithm I commented everythink that is not usefull to compute scenario tree transport matrices
"""


import numpy as np

# Hyperparameters
lambda_sinkhorn = 9
iterations = 20


def sinkhorn_descent(a, b, M, lambda_sinkhorn=100, iterations=100):
    """
    lambda and iterations are hyperparameters,
    iterations could be replaced with a stopping criterion in the futur
    """
    # # Keep track of the precision:
    # E = np.ones(iterations) * 100
    # Exponential of distances
    K = np.exp(-lambda_sinkhorn * M)

    # Remove zeros in a to avoid division by zeros is PRIMORDIAL :
    n = len(a)
    I = a > 0
    # n_I = len(I)
    a = a[I]
    # M = M[I, :]
    K = K[I, :]

    # Reshape b to make it a matrix if it s just one list
    if len(np.shape(b)) == 1:
        b = np.reshape(b, (len(b), 1))

    K_tild = np.diag(np.reciprocal(a)) @ K

    # initialisation of u, an (n x 1) matrix, with
    n_no_null = len(a)
    u = np.ones((n_no_null, 1)) / n_no_null

    # Iterate for u
    # FIXME : see rule of Wolf for the stopping criterion?
    epsilon_sinkhorn = 1
    for iteration in range(iterations):  # epsilon_sinkhorn > precision_sinkhorn : #and iteration < 100:
        # # keep track on convergence
        # u1 = u

        # calcul
        inv_u = K_tild @ (b * np.reciprocal(K.T @ u))
        u = np.reciprocal(inv_u)

        ## convergence:
        # dist = (u1 - u) / np.linalg.norm(u)  # normalized by the 2-norm of u
        # epsilon_sinkhorn = np.sqrt(dist.T @ dist)
        # epsilon_sinkhorn = epsilon_sinkhorn[0, 0]
        # E[iteration] = epsilon_sinkhorn

    # print(iteration, '-th, precision : ', epsilon_sinkhorn, "norm u: ", np.linalg.norm(u))

    # Get v from u
    v = b * np.reciprocal(K.T @ u)

    # # Get alpha, the subgradient of the Sinkhorn distance
    # alpha = np.zeros((len(I), np.shape(b)[1]))
    # alpha_I = -1 / lambda_sinkhorn * np.log(u) + 1 / (n_no_null * lambda_sinkhorn) * sum(np.log(u)) * np.ones(
    #     (n_no_null, 1))
    # alpha[I, :] = alpha_I

    # Get T, the transport matrix
    T = np.zeros((n, len(b)))  # modified from (len(I),len(I))
    u_flat = np.reshape(u, (len(u),))  # needed to use np.diag
    v_flat = np.reshape(v, (len(v),))  # needed to use np.diag
    T_I = np.diag(u_flat) @ K @ np.diag(v_flat)
    T[I, :] = T_I

    # Sinkhorn distance ~ Wasserstein distance
    # D_S = np.trace(M.T @ T_I)

    # Output: the subgradient, the transport matrix, the distance
    return (T) #, D_S) # alpha,  E