import numpy as np

def alpha(w, h, u):
    '''
    Calculates one entry in for the similarity matrix using one
    column from H and one column from U
    '''
    # alpha(h,u) = w^T * [h; u; h * u]
    
    concat = np.c_[h, u, h * u]
    if w.shape != concat.shape:
        raise ValueError("shape of w and [h; u; h * u] do not match")

    # return a scalar
    return np.dot(w, concat)

def similarity(w, H, U):
    '''
    Use alpha function to calculate similarity for each entry

    w: trainable weight vector
    H: context matrix
    U: query matrix
    '''

    sim_matrix = np.zeros((H.shape[0], U.shape[0]))

    for i in range(H.shape[0]):
        # traverse every column in H
        h = H[:,i]
        for j in range(U.shape[0]):
            # traverse every column in U
            u = U[:,j]
            sim_matrix[i][j] = alpha(w, h, u)
    
    return sim_matrix

def context_to_query(S, U):
    '''
    Use similarity and U to calculate Ũ.
    Ũ encapsulates the information about the relevance of each query word to each Context word.

    S: similarity matrix calculated using the similarity() function
    U: query matrix
    '''

    # calculate rowsize softmax of S
    A = np.zeros(S.shape)
    for i in range(S.shape[0]):
        row = S[i]
        A[i] = softmax(row)

    # TODO: double check this
    U_t = np.dot(A, U) # should be H.shape[0] x U.shape[1]
    # if U_t.shape != (H.shape[0], U.shape[1]):
        # raise Exception("something went wrong calculating U_t")

    return U_t

def query_to_context(S, H):
    '''
    S: similarity matrix calculated using the similarity() function
    H: context matrix
    '''
    # get the maximum similarity from each row
    z = np.max(S, axis=1) 

    b = softmax(z)
    h_hat = np.dot(b,H)

    # H.shape[0] or H.shape[1] ? 
    H_hat = np.tile(h_hat, (H.shape[1],1)).T
    return H_hat

def megamerge(H, U_t, H_hat):
    if H.shape != U_t.shape or U_t.shape != H_hat.shape:
        raise ValueError("Incorrect shapes: H=", H.shape, \
                         "U_t=", U_t.shape, "H_hat=", H_hat.shape)
    
    G = [] # 8d x T
    for i in H.shape[0]:
        # column merge
        col_i = beta(H[:,i], U_t[:,i], H_hat[:,i])
        G.append(col_i)
    
    G = np.asarray(G)
    return G

def beta(a,b,c):
    '''
    Used for megamerging
    [a ; b ; a * b ; a * c]
    '''
    return np.c_[a, b, a * b, a * c]
    

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def main(w, H, U):
    '''
    The attention flow layer requires
    w: a weight vector 
    H: context embed matrix
    U: query embed matrix
    '''
    S = similarity(w, H, U)
    U_t = context_to_query(S, U)
    H_hat = query_to_context(S, H)
    G = megamerge(H, U_t, H_hat)