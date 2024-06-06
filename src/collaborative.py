import time

import numpy as np
from scipy import sparse
from scipy.sparse import linalg

def collaborative_filtering(sparse_matrix, sparse_matrix_original, sparse_matrix_test_original, k, baseline = False):

    
    if not baseline:
        print(f'Collaborative Filtering')
    else:
        print(f'Collaborative Filtering with Baseline Approach')

    collaborative_matrix = sparse_matrix_original.toarray()

    # store rows and norm(row)
    row_norm = []
    for r in range(sparse_matrix.shape[0]):
        row_norm.append(linalg.norm(sparse_matrix.getrow(r)))
    row_norm = np.array(row_norm).reshape((sparse_matrix.shape[0], 1))

    # store product of norm every 2 rows in users in row_norm_products
    row_norm_products = np.dot(row_norm, np.transpose(row_norm))

    # store dot product of every 2 rows in dot_products
    dot_products = sparse_matrix.dot(sparse_matrix.transpose()).toarray()

    # get indices of upper triangular matrix
    iu1 = np.triu_indices(dot_products.shape[0], 1)
    rowu_indices, colu_indices = iu1

    # store similarities for each user-user tuple
    similarity = np.diag(np.array([-2 for x in range(sparse_matrix.shape[0])], dtype = np.float32))
    similarity[rowu_indices, colu_indices] = dot_products[rowu_indices,colu_indices]/row_norm_products[rowu_indices, colu_indices]
    similarity[np.isnan(similarity)] = 0.0

    # copy upper triangular values to lower triangle
    similarity.T[rowu_indices, colu_indices] = similarity[rowu_indices, colu_indices]

    # store indices of closest k users for every user
    neighbourhood = np.zeros((sparse_matrix.shape[0], k), dtype = np.int32)
    for i in range(sparse_matrix.shape[0]):
        neighbourhood[i] = similarity[i,:].argsort()[-k:][::-1]

    # store similarity of user u with its k neighbours
    row_indices, col_indices = np.indices(neighbourhood.shape)
    similarity_user_neighbour = np.zeros(neighbourhood.shape, dtype = np.float32)
    similarity_user_neighbour[row_indices, col_indices] = similarity[row_indices,neighbourhood[row_indices,col_indices]]

    row_test_indices, col_test_indices = sparse_matrix_test_original.nonzero()

    if not baseline:
        for i, (r, c) in enumerate(zip(row_test_indices, col_test_indices)):
            collaborative_matrix[r, c] = np.dot(similarity_user_neighbour[r], np.array([collaborative_matrix[j,c] for j in neighbourhood[r]]))/(similarity_user_neighbour[r].sum())
    else:
        # find mean of whole original sparse matrix
        mean = sparse_matrix_original.sum()/(sparse_matrix_original!=0).sum()
        # find mean rating for every user
        user_mean = (np.squeeze(np.array(sparse_matrix_original.sum(axis = 1)/(sparse_matrix_original!=0).sum(axis = 1)))).tolist()
        # find mean rating of every movie
        movie_mean = (np.squeeze(np.array(sparse_matrix_original.sum(axis = 0)/(sparse_matrix_original!=0).sum(axis = 0)))).tolist()

        row_indices, col_indices = sparse_matrix_original.nonzero()
        data = np.array([user_mean[r] + movie_mean[c] - mean for (r,c) in zip(row_indices, col_indices)])
        baseline_matrix = sparse.coo_matrix((data, (row_indices,col_indices)),shape = sparse_matrix_original.shape).toarray()

        for i, (r, c) in enumerate(zip(row_test_indices, col_test_indices)):
            collaborative_matrix[r, c] = baseline_matrix[r,c] + np.dot(similarity_user_neighbour[r], np.array([collaborative_matrix[j,c]-baseline_matrix[j,c] for j in neighbourhood[r]]))/(similarity_user_neighbour[r].sum())

    
    collaborative_matrix[np.isnan(collaborative_matrix)] = 0.0
    return collaborative_matrix
