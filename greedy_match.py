import numpy as np
from scipy.sparse import csr_matrix
import time

def greedy_match(X):
    
    m, n = X.shape
    N = m * n
    t0 = time.process_time()

    x = X.flatten()
    minSize = min(m, n)
    usedRows = np.zeros(m)
    usedCols = np.zeros(n)

    maxList = np.zeros(minSize)
    row = np.zeros(minSize)
    col = np.zeros(minSize)

    sorted_indices = np.argsort(x)[::-1]
    matched = 1
    index = 0

    while matched <= minSize:
        ipos = sorted_indices[index]  # position in the original vectorized matrix
        jc = ipos // m
        ic = ipos - (jc * m)

        if ic == 0:
            ic = 1

        if usedRows[ic-1] != 1 and usedCols[jc-1] != 1:
            row[matched-1] = ic
            col[matched-1] = jc
            maxList[matched-1] = x[index]
            usedRows[ic-1] = 1
            usedCols[jc-1] = 1
            matched += 1

        index += 1

    data = np.ones(minSize)
    M = csr_matrix((data, (row-1, col-1)), shape=(m, n))
    dt = time.process_time() - t0

    return M, dt