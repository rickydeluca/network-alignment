"""
The following code is an adapation in Python of the
FINAL algorithm developed by Si Zhang originally written
in MATLAB code.

The origianl code can be found at the following GitHub Repository:
https://github.com/sizhang92/FINAL-KDD16
"""

import torch
import torch.sparse as sp

def FINAL(A1, A2, H, alpha, maxiter, tol, N1=None, N2=None, E1=None, E2=None):
    """
    Description:
        The algorithm is the generalized attributed network alignment algorithm.
        The algorithm can handle the cases no matter node attributes and/or edge
        attributes are given. If no node attributes or edge attributes are given,
        then the corresponding input variable of the function is empty, e.g.,
        N1 = [], E1 = {}.
        The algorithm can handle either numerical or categorical attributes
        (feature vectors) for both edges and nodes.

        The algorithm uses cosine similarity to calculate node and edge feature
        vector similarities. E.g., sim(v1, v2) = <v1, v2>/(||v1||_2*||v2||_2).
        For categorical attributes, this is still equivalent to the indicator
        function in the original published paper.

    Input:
        A1, A2: Input adjacency matrices with n1, n2 nodes
        N1, N2: Node attributes matrices, N1 is an n1K matrix, N2 is an n2K
        matrix, each row is a node, and each column represents an
        attribute. If the input node attributes are categorical, we can
        use one hot encoding to represent each node feature as a vector.
        And the input N1 and N2 are still n1K and n2K matrices.
        E.g., for node attributes as countries, including USA, China, Canada,
        if a user is from China, then his node feature is (0, 1, 0).
        If N1 and N2 are emtpy, i.e., N1 = [], N2 = [], then no node
        attributes are input.

        E1, E2: a L1 cell, where E1{i} is the n1n1 matrix and nonzero entry is
        the i-th attribute of edges. E2{i} is same. Similarly, if the
        input edge attributes are categorical, we can use one hot
        encoding, i.e., E1{i}(a,b)=1 if edge (a,b) has categorical
        attribute i. If E1 and E2 are empty, i.e., E1 = {} or [], E2 = {}
        or [], then no edge attributes are input.

        H: a n2*n1 prior node similarity matrix, e.g., degree similarity. H
        should be normalized, e.g., sum(sum(H)) = 1.
        alpha: decay factor
        maxiter, tol: maximum number of iterations and difference tolerance.

    Output:
        S: an n2*n1 alignment matrix, entry (x,y) represents to what extend node-
        x in A2 is aligned to node-y in A1

    Reference:
        Zhang, Si, and Hanghang Tong. "FINAL: Fast Attributed Network Alignment."
        Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining. ACM, 2016.
    """

    n1 = A1.size(0)
    n2 = A2.size(0)

    # If no node attributes input, then initialize as a vector of 1
    # so that all nodes are treated to have the same attributes, which
    # is equivalent to no given node attributes.
    if N1 is None: 
        N1 = torch.ones(n1, 1)
    
    if N2 is None:
        N2 = torch.ones(n2, 1)

    # If no edge attributes are input, initialize as a cell with 1 element,
    # which is the same as the adjacency matrix but with nonzero entries in
    # the adjacency matrix set to 1, treating all edges as having the same
    # attributes. This is equivalent to no given edge attributes.
    if E1 is None:
        E1 = [A1]
        E1[0][A1 > 0] = 1
    
    if E2 is None:
        E2 = [A2]
        E2[0][A2 > 0] = 1

    K = N1.size(1)
    L = len(E1)
    T1 = sp.eye(n1, n1)
    T2 = sp.eye(n2, n2)

    # Normalize edge feature vectors
    for l in range(L):
        T1 += E1[l] ** 2  # calculate ||v1||_2^2
        T2 += E2[l] ** 2  # calculate ||v2||_2^2
    T1 = T1 ** -0.5
    T2 = T2 ** -0.5
    for l in range(L):
        E1[l] = E1[l] * T1  # normalize each entry by vector norm T1
        E2[l] = E2[l] * T2  # normalize each entry by vector norm T2

    # Normalize node feature vectors
    K1 = torch.norm(N1, dim=1).pow(-0.5)
    K1[K1 == float('inf')] = 0
    K2 = torch.norm(N2, dim=1).pow(-0.5)
    K2[K2 == float('inf')] = 0
    N1 = N1 * K1.view(-1, 1)  # normalize the node attribute for A1
    N2 = N2 * K2.view(-1, 1)  # normalize the node attribute for A2

    # Compute node feature cosine cross-similarity
    N = torch.zeros(n1 * n2, 1)
    for k in range(K):
        N += torch.kron(N1[:, k], N2[:, k])  # compute N as a kronecker similarity

    # Compute the Kronecker degree vector
    d = torch.zeros(n1 * n2, 1)
    for l in range(L):
        for k in range(K):
            d += torch.kron((E1[l] * A1) @ N1[:, k], (E2[l] * A2) @ N2[:, k])
    D = N * d
    DD = D.pow(-0.5)
    DD[D == 0] = 0  # define inf as 0

    # fixed-point solution
    q = DD * N
    h = H.view(-1)
    s = h.clone()

    for i in range(maxiter):
        # print(f'iteration {i+1}')
        prev = s.clone()
        M = (q * s).reshape(n2, n1)
        S = torch.zeros(n2 * n1, 1)
        for l in range(L):
            S += sp.mm(sp.mm(E2[l], A2), sp.mm(M, E1[l] * A1))  # calculate the consistency part
        s = (1 - alpha) * h + alpha * (q.to_dense() * S.view(-1, 1))  # add the prior part
        diff = torch.norm(s - prev)

        # print(f'Time for iteration {i+1}: diff = {100 * diff:.5f}')
        if diff < tol:  # if converge
            break

    S = s.reshape(n2, n1)  # reshape the similarity vector to a matrix
    return S
