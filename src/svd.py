import time

import numpy as np
from scipy.sparse import linalg
from scipy.sparse.linalg import LinearOperator, eigsh



def svd_sparse(sparse_matrix, no_eigen_values):

    def transpose(matrix):
        return matrix.transpose()

    def matvec_XH_X(vector):
        return XH_dot(X_dot(vector))

    n, m = sparse_matrix.shape
    X_dot = sparse_matrix.dot
    XH_dot = transpose(sparse_matrix).dot

    XH_X = LinearOperator(
        matvec=matvec_XH_X,
        dtype=sparse_matrix.dtype,
        shape=(min(n, m), min(n, m))
    )

    eigvals, eigvecs = eigsh(XH_X, k=no_eigen_values)
    eigvals = np.maximum(eigvals.real, 0)

    # Create sigma diagonal matrix
    sigma = np.sqrt(eigvals)
    s = np.zeros_like(eigvals)
    s[:no_eigen_values] = sigma

    u = X_dot(eigvecs) / sigma
    vt = transpose(eigvecs)

    return u, s, vt

def svd_retain_energy(sparse_matrix, no_eigen_values, energy = 1):
    u, s, vt = svd_sparse(sparse_matrix, no_eigen_values)
    s_squared_sum = np.square(s).sum()		# sum of square of all eigen values (diagnol elements in s)

    for i in range(s.shape[0]):
        if np.square(s[i:]).sum()<(energy*s_squared_sum):
            break
    i -= 1

    return np.delete(u, np.s_[:i], 1), s[i:], np.delete(vt, np.s_[:i], 0)


def svd(sparse_matrix, no_eigen_values, energy = 1):

    start = time.time()
    print(f'---- SVD with {energy * 100}% energy ----')

    u,s,vt = svd_retain_energy(sparse_matrix, no_eigen_values, energy)
    svd_matrix = np.dot(np.dot(u,np.diag(s)), vt)

    print('time taken ' + '{0:.2f}'.format(time.time() - start) + ' secs.')
    return svd_matrix
