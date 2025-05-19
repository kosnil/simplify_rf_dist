from scipy.sparse import csr_matrix, lil_matrix
import numpy as np
from joblib import Parallel, delayed
import scipy.sparse as scs


def top_n_idx_sparse(matrix: csr_matrix, n: int) -> tuple:
    """
    Return index of top n values in each row of a sparse matrix.
    Parameters:
    - matrix: csr_matrix. Shape (n_test, n_train).
    - n: int, number of top values to return
    Returns:
    - top_n_idx: list of lists, each containing the indices of the top n values in each row
    - top_n_dataidx: list of lists, each containing the indices of the top n values in each row
    """
    top_n_idx = []
    top_n_dataidx = []
    for le, ri in zip(matrix.indptr[:-1], matrix.indptr[1:]):
        n_row_pick = min(n, ri - le)
        row_n_data_idx = le + np.argpartition(matrix.data[le:ri], -n_row_pick)[-n_row_pick:]
        top_n_idx.append(matrix.indices[row_n_data_idx])
        top_n_dataidx.append(row_n_data_idx)

    return top_n_idx, top_n_dataidx


def sparsify_csr(w_csr: csr_matrix, top_idx: int, top_dataidx: int, return_sum=False):
    """Sparsify a csr matrix by keeping only the top values in each row.
    Parameters:
    - w_csr (csr_matrix): Matrix to be 'sparsified' in the Topk-sense. Shape (n_test, n_train).
    - top_idx (list): List of lists, each containing the indices of the top values in each row.
    - top_dataidx (list): List of lists, each containing the indices of the top values in each row.
    - return_sum (bool): If True, return the sum of the top values in each row.
    Returns:
    - new_w (csr_matrix): Sparsified matrix with the same shape as w_csr.
    - w_sums (np.array): Array of sums of the top values in each row. Only returned if return_sum is True.
    """

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


def sparse_cumsum(arr: csr_matrix) -> csr_matrix:
    """Compute the cumulative sum of a sparse matrix along the rows.
    This function sums only the non-zero elements of the sparse matrix.
    Thus, it is not a simple cumulative sum but a cumulative sum of the non-zero elements and you cannot use this as a replacement for np.cumsum.
    Parameters:
    - arr: csr_matrix
    Returns:
    - arr: csr_matrix with cumulative sums
    """
    arr = arr.copy()

    indptr = arr.indptr
    data = arr.data
    for i in range(arr.shape[0]):
        st = indptr[i]
        en = indptr[i + 1]
        np.cumsum(data[st:en], out=data[st:en])

    return arr


def sort_lil_row(lil_row: lil_matrix, idx: int) -> lil_matrix:
    """
    Sort a row of a lil_matrix according to the given indices.
    Parameters:
    - lil_row: lil_matrix, a single row of a lil_matrix
    - idx: list, indices to sort the row
    Returns:
    - lil_row: lil_matrix, sorted row
    """
    return lil_row[:, idx]


def sort_lil(w_all_sparse, idx_sort) -> csr_matrix:
    """
    Sort a sparse matrix in the lil format according to the given indices.
    Parameters:
    - w_all_sparse (sparse matrix): sparse matrix to be sorted. Shape (n_test, n_train).
    - idx_sort (list): indices to sort the matrix
    Returns:
    - w_all_sparse (csr_matrix): sorted sparse matrix
    """
    # Convert to lil format for efficient row-wise operations
    w_all_sparse = w_all_sparse.tolil()

    results = Parallel(n_jobs=-1)(
        delayed(sort_lil_row)(w_all_sparse[i], idx_sort) for i in range(w_all_sparse.shape[0]))
    w_all_sparse = scs.vstack(results).tocsr()

    return w_all_sparse
