from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
from joblib import Parallel, delayed
import scipy.sparse as scs


def top_n_idx_sparse(matrix, n):
    """
    Return index of top n values in each row of a sparse matrix.
    Parameters:
    - matrix: csr_matrix
    """
    top_n_idx = []
    top_n_dataidx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        row_n_data_idx = le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]
        top_n_idx.append(matrix.indices[row_n_data_idx])
        top_n_dataidx.append(row_n_data_idx)

    return top_n_idx, top_n_dataidx


def sparsify_csr(w_csr, top_idx, top_dataidx, return_sum=False):
    """Sparsify a csr matrix by keeping only the top values in each row."""
    new_w = lil_matrix(w_csr.shape, dtype=np.float32)

    w_sums = [] if return_sum else None

    for i in range(new_w.shape[0]):
        w_sum = w_csr.data[top_dataidx[i]].sum()
        new_w[i, top_idx[i]] = w_csr.data[top_dataidx[i]] / w_sum

        if return_sum:
            w_sums.append(w_sum)

    if return_sum:
        return new_w.tocsr(), np.array(w_sums)

    return new_w.tocsr()


def sparse_cumsum(arr):
    arr = arr.copy()

    indptr = arr.indptr
    data = arr.data
    for i in range(arr.shape[0]):
        st = indptr[i]
        en = indptr[i + 1]
        np.cumsum(data[st:en], out=data[st:en])

    return arr


def sort_lil_row(lil_row, idx):
    return lil_row[:, idx]


def sort_lil(w_all_sparse, idx_sort):

    w_all_sparse = w_all_sparse.tolil()

    results = Parallel(n_jobs=-1)(
        delayed(sort_lil_row)(w_all_sparse[i], idx_sort) for i in range(w_all_sparse.shape[0]))
    w_all_sparse = scs.vstack(results).tocsr()

    return w_all_sparse
