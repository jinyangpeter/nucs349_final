import numpy as np 

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """

    M = X.shape[0]
    N = Y.shape[0]
    D = np.zeros([M, N])

    for i in range(M):
        for j in range(N):
            D[i, j] = np.sqrt(np.dot(X[i] - Y[j], X[i] - Y[j]))

    return D




def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """


    M = X.shape[0]
    N = Y.shape[0]
    D = np.zeros([M, N])

    for i in range(M):
        for j in range(N):
            temp = np.abs(X[i] - Y[j])
            D[i, j] = np.sum(temp)

    return D


def cosine_distances(X, Y):
    """Compute Cosine distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Cosine distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Cosine distances between rows of X and rows of Y.
    """

    M = X.shape[0]
    N = Y.shape[0]
    D = np.zeros([M, N])

    for i in range(M):
        for j in range(N):
            n_1 = np.sqrt(np.dot(X[i], X[i]))
            n_2 = np.sqrt(np.dot(Y[j], Y[j]))
            # This yield the cosine value of the two vectors, which range from 1 to -1
            D[i, j] = np.dot(X[i], Y[j]) / (n_1 * n_2)

    # Larger value (closer to 1) means the two vectors have similar direction, so they
    # should have smaller distance
    return 1 - D
