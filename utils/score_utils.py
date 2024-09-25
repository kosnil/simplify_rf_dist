import numpy as np
import statsmodels.api as sm
import logging
import time
import multiprocessing
from joblib import Parallel, delayed
import gc

from utils.sparse_utils import *

from tqdm import tqdm

# def gini(x):
#     # The rest of the code requires numpy arrays.
#     x = np.asarray(x)

#     sorted_x = np.sort(x)
#     n = len(x)
#     cumx = np.cumsum(sorted_x, dtype=float)
#     # The above formula, with all weights equal to 1 simplifies to:
#     return (n + 1 - 2 * np.sum(cumx) / cumx[-1]) / n


#
def gini_mat(x):
    """
    Calculate the Gini coefficient for a matrix of values, based on https://stackoverflow.com/questions/48999542/more-efficient-weighted-gini-coefficient-in-python
    Parameters:
    - x (array-like): The input matrix. 
    
    Returns:
    - array-like: The Gini coefficient for each row of the matrix.
    """

    sorted_x = np.sort(x, axis=1)
    n = x.shape[1]
    cumx = np.cumsum(sorted_x, dtype=float, axis=1)
    # The above formula, with all weights equal to 1 simplifies to:
    return (n + 1 - 2 * np.sum(cumx, axis=1)) / n


def simple_dm(l1, l2):
    """
    Simple implementation of Diebold-Mariano test.
    Paramters:
    - l1 (array-like): The first list of errors.
    - l2 (array-like): The second list of errors.
    
    Returns:
    - res (statsmodels.regression.linear_model.RegressionResultsWrapper): The results of the Diebold-Mariano test.
    """
    d = l1 - l2
    mod = sm.OLS(d, np.ones(len(d)))
    res = mod.fit().get_robustcov_results(cov_type='HAC', maxlags=1)
    return res


def calc_r2(y_true, y_pred, y_train=None):
    """
    Calculate the R-squared (coefficient of determination) for a regression model.

    Parameters:
    - y_true (array-like): The true target values.
    - y_pred (array-like): The predicted target values.
    - y_train (array-like, optional): The target values used for training the model. 
      If provided, it will be used to calculate the variance of y_true.

    Returns:
    - r2 (float): The R-squared value, which represents the proportion of the variance in the dependent variable 
      that is predictable from the independent variables.

    """

    res_mean = ((y_true - y_pred)**2).mean()

    if y_train is None:
        y_true_var = y_true.var()
    else:
        y_train_mean = np.mean(y_train)
        y_true_var = ((y_true - y_train_mean)**2).mean()

    r2 = 1 - (res_mean / y_true_var)

    return r2


def quantile_score(y_true, y_pred, alpha):
    """
    Calculate the quantile score.

    Parameters:
    - y_true: The true values.
    - y_pred: The predicted values.
    - alpha: The quantile level.

    Returns:
    - The quantile score.

    """

    diff = y_true - y_pred
    indicator = (diff >= 0).astype(diff.dtype)
    loss = indicator * alpha * diff + (1 - indicator) * (1 - alpha) * (-diff)

    return 2 * loss


def se(y_true, y_pred):
    """
    Calculates the squared error (SE) between the true values and the predicted values.

    Parameters:
    - y_true (array-like): The true values.
    - y_pred (array-like): The predicted values.

    Returns:
    - array-like
        The SE between the true values and the predicted values.
    """
    return (y_true - y_pred)**2


def ae(y_true, y_pred):
    """
    Calculates the absolute error (AE) between the true values and the predicted values.

    Parameters:
    - y_true (array-like): The true values.
    - y_pred (array-like): The predicted values.

    Returns:
    - array-like
        The AE between the true values and the predicted values.
    """
    return np.abs(y_true - y_pred)


def mse(y_true, y_pred):
    """
    Calculates the mean squared error (MSE) between the true values and the predicted values.

    Parameters:
    - y_true (array-like): The true values.
    - y_pred (array-like): The predicted values.

    Returns:
    - float: The MSE between the true values and the predicted values.
    """
    return se(y_true, y_pred).mean()


def mae(y_true, y_pred):
    """
    Calculate the mean absolute error (MAE) between the true values and the predicted values.

    Parameters:
    - y_true: numpy array or list
        The true values.
    - y_pred: numpy array or list
        The predicted values.

    Returns:
    - float
        The mean absolute error between the true values and the predicted values.
    """
    return ae(y_true, y_pred).mean()


def fill_zeros_with_last(arr):
    """
    Fills zeros in the input array with the last non-zero value.

    Parameters:
    arr (numpy.ndarray): The input array.

    Returns:
    numpy.ndarray: The modified array with zeros replaced by the last non-zero value.
    """
    prev = np.arange(len(arr))
    prev[arr == 0] = 0
    prev = np.maximum.accumulate(prev)
    return arr[prev]


# def crps_sample_sparse(y, dat, w, dat_ordered=True, order=None):

#     y = y.astype(np.float32)
#     dat = dat.astype(np.float32)

#     if dat_ordered:
#         x = dat
#     else:
#         if order is None:
#             order = np.argsort(dat)
#         x = dat[order]

#     score_arr = np.zeros((len(y)), dtype=np.float32)

#     for i in range(w.shape[0]):
#         # wi = w[i].toarray().squeeze()
#         wi = w[i]
#         yi = y[i]
#         # p = np.cumsum(wi)
#         p = fill_zeros_with_last(sparse_cumsum(wi).toarray().squeeze())
#         wi = wi.toarray().squeeze()
#         # a = np.asarray(p - 0.5 * wi).squeeze()
#         a = (p - 0.5 * wi)

#         indicator = (yi < x).astype(x.dtype)
#         score_arr[i] = (wi * (indicator - a) * (x - yi)).sum()

#     return 2. * score_arr


def crps_sample_sparse2(y, dat, w, dat_ordered=True, order=None):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for a given set of sparse samples.
    Improved version of crps_sample_sparse with better performance (only iterates over non-zero elements).
    Based on the R-package 'scoringRules'.

    Parameters:
    - y (ndarray): Array of true values.
    - dat (ndarray): Array of predicted values.
    - w (ndarray): Array of weights.
    - dat_ordered (bool): Flag indicating if the predicted values are already ordered.
    - order (ndarray): Array of indices to order the predicted values.

    Returns:
    - ndarray: Array of CRPS scores.

    """

    if dat_ordered:
        x = dat
    else:
        if order is None:
            order = np.argsort(dat)
        x = dat[order]

    score_arr = np.zeros((len(y)), dtype=np.float32)

    for i in range(w.shape[0]):

        nonzero_idxs = w[i].indices
        wi = w[i].data
        yi = y[i]
        x_sparse = x[nonzero_idxs]

        p = np.cumsum(wi.data)

        a = (p - 0.5 * wi)

        indicator = (yi < x_sparse).astype(x.dtype)
        score_arr[i] = (wi * (indicator - a) * (x_sparse - yi)).sum()

    return 2. * score_arr


# def crps_for_loop(wi, yi, x):

#     wi = wi.toarray().squeeze()
#     p = np.cumsum(wi)
#     a = (p - 0.5 * wi)

#     indicator = (yi < x).astype(x.dtype)
#     return (wi * (indicator - a) * (x - yi)).sum()

# def crps_sample_sparse_parallel(y, dat, w, dat_ordered=True, order=None):

#     y = y.astype(np.float32)
#     dat = dat.astype(np.float32)

#     if dat_ordered:
#         x = dat
#     else:
#         if order is None:
#             order = np.argsort(dat)
#         x = dat[order]

#     # score_arr = np.zeros((len(y)))

#     results = Parallel(n_jobs=-1)(delayed(crps_for_loop)(w[i], y[i], x) for i in range(w.shape[0]))

#     score_arr = np.array(results)

#     return 2. * score_arr


def crps_sample(y, dat, w, return_mean=True):
    """
    Calculate the Continuous Ranked Probability Score (CRPS) for given samples.
    Implementation based on the R-package 'scoringRules'.

    Parameters:
    - y (numpy.ndarray): Array of true values.
    - dat (numpy.ndarray): Array of predicted values.
    - w (numpy.ndarray): Array of weights.
    - return_mean (bool): Flag indicating whether to return the mean CRPS score. Default is True.

    Returns:
    - float or numpy.ndarray: The CRPS score(s). If return_mean is True, returns a single float representing the mean CRPS score.
                              If return_mean is False, returns an array of CRPS scores, one for each sample.

    """

    y = y.astype(np.float32)
    dat = dat.astype(np.float32)

    order = np.argsort(dat)
    x = dat[order]

    score_arr = np.zeros((len(y)))

    for i in range(w.shape[0]):
        wi = w[i][order]
        yi = y[i]
        p = np.cumsum(wi)
        P = p[-1]
        a = (p - 0.5 * wi) / P

        # score = 2 / P * np.sum(wi * (np.where(yi < x, 1. , 0.) - a) * (x - yi))
        indicator = (yi < x).astype(x.dtype)
        score = 2 / P * (wi * (indicator - a) * (x - yi)).sum()

        score_arr[i] = score

    if return_mean:
        return score_arr.mean()

    return score_arr


def crps_sample_fast(y, dat, w):
    """
    Slightly faster version of crps_sample adapted to our specific usecase and structure of w.
    """

    order = np.argsort(dat)
    x = dat[order]

    sample_len = len(y)

    score_arr = np.zeros((sample_len))

    for i in range(sample_len):
        wi = w[i][order]
        wi_nonzero = wi.astype(bool)
        wi = wi[wi_nonzero]
        x_sparse = x[wi_nonzero]
        yi = y[i]
        p = np.cumsum(wi)
        a = p - 0.5 * wi

        indicator = (yi < x_sparse).astype(x.dtype)
        score_arr[i] = (wi * (indicator - a) * (x_sparse - yi)).sum()

    return 2 * score_arr


def crps_sample_unconditional(y, dat):
    """
    Unconditional version of CRPS calculation,i.e. we're not considering the calculated weights. 
    Instead, all training samples are weighted equally.
    See crps_sample for paramter details.
    """

    order = np.argsort(dat)
    x = dat[order]

    sample_len = len(y)

    score_arr = np.zeros((sample_len))

    wi = np.repeat(1 / len(dat), len(dat))
    p = np.cumsum(wi)
    a = (p - 0.5 * wi)

    for i in range(len(y)):

        yi = y[i]

        # score = 2 / P * np.sum(wi * (np.where(yi < x, 1. , 0.) - a) * (x - yi))
        indicator = (yi < x).astype(x.dtype)
        score = (wi * (indicator - a) * (x - yi)).sum()

        score_arr[i] = score

    return 2 * score_arr


def calc_metrics_topk(k, w_hat, y_train, y_test, verbose=False):
    """
    Calculate various metrics for top-k predictions for specific k.

    Parameters:
    k (int): The number of top elements to consider.
    w_hat (array-like): The weight matrix.
    y_train (array-like): The training labels.
    y_test (array-like): The test labels.

    Return
    tuple: A tuple containing the following metrics for the given samples:
        - r2k (float): R^2.
        - se_test (float): SE.
        - ae_test (float): AE.
        - crps_test (float): CRPS.
        - corr_test (float): Correlation between predicted and true values.
        - cov_test (float): The covariance between predicted and true values.
        - var_y_tk (float): The variance of forecast.
        - bias_y_tk (float): The bias of forecast.
        - w_topk_sums (array-like): Sum of remaining top k weights before normalization.
        - y_tk (array-like): Forecast.

    Raises:
    Exception: If an error occurs during the calculation.
    """

    try:
        y_train_as = np.argsort(y_train)
        y_train_s = y_train[y_train_as]

        row_idx = np.argpartition(w_hat, -k, axis=1)[:, -k:]
        i = np.indices(row_idx.shape)[0]

        w_tmp = np.zeros(w_hat.shape, dtype=np.float32)
        if isinstance(w_hat, np.ndarray):
            w_tmp[i, row_idx] = w_hat[i, row_idx]
        else:
            w_tmp[i, row_idx] = w_hat[:][i, row_idx]
        w_topk_sums = w_tmp.sum(axis=1)
        w_k = w_tmp / w_topk_sums[:, None]

        y_tk = w_k @ y_train

        ecdfs = np.cumsum(w_k[:, y_train_as], axis=1)
        q_idxs = np.argmax(ecdfs >= .5, axis=1)
        y_tk_median = (y_train_s[q_idxs - 1] + y_train_s[q_idxs]) * .5

        r2k = calc_r2(y_test, y_tk, y_train=y_train)

        se_test = se(y_test, y_tk)

        ae_test = ae(y_test, y_tk_median)

        crps_test = crps_sample(y_test, y_train, w_k, return_mean=False)

        corr_test = np.corrcoef(y_test, y_tk)[0, 1]

        cov_test = np.cov(y_test, y_tk)[0, 1]

        bias_y_tk = np.mean(y_tk) - np.mean(y_test)

        var_y_tk = np.var(y_tk)

    except Exception as e:
        print(e)
        logging.error(f"Error in calculate_metrics: {e}")

    return (r2k, se_test, ae_test, crps_test, corr_test, cov_test, var_y_tk, bias_y_tk, w_topk_sums, y_tk)


def calc_metrics_topk_sparse(k, w_hat, y_train, y_test):
    """
    Calculate various metrics for sparse top-k predictions for specific k.

    Parameters:
    k (int): The number of top elements to consider.
    w_hat (sparse matrix): The weight matrix.
    y_train (array-like): The training labels.
    y_test (array-like): The test labels.

    Return
    tuple: A tuple containing the following metrics for the given samples:
        - r2k (float): R^2.
        - se_test (float): SE.
        - ae_test (float): AE.
        - crps_test (float): CRPS.
        - corr_test (float): Correlation between predicted and true values.
        - cov_test (float): The covariance between predicted and true values.
        - var_y_tk (float): The variance of forecast.
        - bias_y_tk (float): The bias of forecast.
        - w_topk_sums (array-like): Sum of remaining top k weights before normalization.
        - y_tk (array-like): Forecast.

    Raises:
    Exception: If an error occurs during the calculation.
    """

    try:
        top_idx, top_dataidx = top_n_idx_sparse(w_hat, k)
        w_k, w_topk_sums = sparsify_csr(w_hat, top_idx, top_dataidx, return_sum=True)

        del w_hat
        del top_idx, top_dataidx

        y_tk = w_k @ y_train

        ecdfs = sparse_cumsum(w_k)
        q_idxs = np.asarray((ecdfs >= .5).argmax(axis=1)).squeeze()
        y_tk_median = (y_train[q_idxs - 1] + y_train[q_idxs]) * .5

        del ecdfs, q_idxs

        ae_test = ae(y_test, y_tk_median).astype(np.float32)

        del y_tk_median

        crps_test = crps_sample_sparse2(y_test, y_train, w_k, dat_ordered=True).astype(np.float32)

        del w_k

        r2k = calc_r2(y_test, y_tk, y_train=y_train)

        se_test = se(y_test, y_tk).astype(np.float32)

        corr_test = np.corrcoef(y_test, y_tk)[0, 1]

        cov_test = np.cov(y_test, y_tk)[0, 1]

        bias_y_tk = np.mean(y_tk) - np.mean(y_test)

        var_y_tk = np.var(y_tk)

        return (r2k, se_test, ae_test, crps_test, corr_test, cov_test, var_y_tk, bias_y_tk, w_topk_sums, y_tk)

    except Exception as e:
        print(e)
        logging.error(f"Error in calculate_metrics: {e}")


def topk_looper_sparse(X_test, y_train, y_test, rf, w_hat, k_max=100, k_stepsize=1, verbose=False):
    """
    Calculate various metrics for a Topk RF model for a range of values for k.

    Parameters:
    - X_test (array-like): Test data.
    - y_train (array-like): Training labels.
    - y_test (array-like): Test labels.
    - rf (object): RF model.
    - w_hat (array-like): Weight matrix.
    - k_max (int, optional): Maximum value of k for top-k loop. Defaults to 100.
    - k_stepsize (int, optional): Step size for k in top-k loop. Defaults to 1.
    - verbose (bool, optional): Whether to print verbose output. Defaults to False.

    Returns:
    - results (dict): Dictionary containing various metrics calculated during the top-k loop.
        - 'r2' (array-like): R^2 values.
        - 'corr' (array-like): Correlation coefficients (between forecast and true values).
        - 'cov' (array-like): Covariance values (between forecast and true values).
        - 'se' (array-like): SE values.
        - 'ae' (array-like): AE values.
        - 'crps' (array-like): CRPS values.
        - 'var' (array-like): Variance (of forecast) values.
        - 'bias' (array-like): Bias (of forecast) values.
        - 'pred' (array-like): Predicted values.
        - 'topk_sums' (array-like): Sum of remaining weights for top k values before normalization.
    """

    all_r2 = []
    all_corr = []
    all_cov = []
    all_se = []
    all_ae = []
    all_crps = []
    all_var = []
    all_bias = []
    all_w_topk_sums = []
    all_pred = []

    y_test_mean = np.mean(y_test)

    idx_sort = np.argsort(y_train)

    y_train = y_train[idx_sort]

    y_hat = rf.predict(X_test)
    y_hat_med = rf.sparse_quantile_predict(q=.5, w_all=w_hat)

    print("Done Predicting. Calculating Metrics for full RF...")

    r2_full_test = calc_r2(y_test, y_hat, y_train=y_train)
    se_full_test = se(y_test, y_hat)
    ae_full_test = ae(y_test, y_hat_med)
    crps_full_test = crps_sample_sparse2(y_test, y_train, w_hat, dat_ordered=True)
    cov_full_test = np.cov(y_test, y_hat)[0, 1]
    corr_full_test = np.corrcoef(y_test, y_hat)[0, 1]

    y_hat_var = np.var(y_hat)
    y_hat_mean = np.mean(y_hat)

    del y_hat_med, y_hat

    print("Starting Topk Loop...")

    t0 = time.time() if verbose else None

    data_sizes = []
    for i in range(w_hat.shape[0]):
        data_sizes.append(w_hat[i].data.shape[0])

    n_jobs = -1 if np.mean(data_sizes) < 1000 else 12

    if w_hat.shape[1] > 500_000:
        n_jobs = 12

    del data_sizes

    gc.collect()
    # k, w_hat, y_train, y_test
    results = Parallel(n_jobs=n_jobs)(
        delayed(calc_metrics_topk_sparse)(i, w_hat, y_train, y_test) for i in range(1, k_max + 1, k_stepsize))

    t1 = time.time() if verbose else None

    if verbose:
        print(f"Time taken: {t1 - t0:.2f}s")

    for result in results:
        r2k, se_test, ae_test, crps_test, corr_test, cov_test, var_y_tk, bias_y_tk, w_topk_sums, y_tk = result
        all_r2.append(r2k)
        all_se.append(se_test)
        all_ae.append(ae_test)
        all_crps.append(crps_test)
        all_corr.append(corr_test)
        all_cov.append(cov_test)
        all_var.append(var_y_tk)
        all_bias.append(bias_y_tk)
        all_w_topk_sums.append(w_topk_sums)
        all_pred.append(y_tk)

    all_r2.append(r2_full_test)
    all_se.append(se_full_test)
    all_ae.append(ae_full_test)
    all_crps.append(crps_full_test)
    all_corr.append(corr_full_test)
    all_cov.append(cov_full_test)
    all_var.append(y_hat_var)
    all_bias.append(y_hat_mean - y_test_mean)

    all_r2 = np.array(all_r2)
    all_corr = np.array(all_corr)
    all_cov = np.array(all_cov)
    all_se = np.array(all_se)
    all_ae = np.array(all_ae)
    all_crps = np.array(all_crps)
    all_var = np.array(all_var)
    all_bias = np.array(all_bias)
    all_w_topk_sums = np.array(all_w_topk_sums)

    results = {
        'r2': all_r2,
        'corr': all_corr,
        'cov': all_cov,
        'se': all_se,
        'ae': all_ae,
        'crps': all_crps,
        'var': all_var,
        'bias': all_bias,
        'pred': all_pred,
        'topk_sums': all_w_topk_sums
    }

    return results


def topk_looper(X_test, y_train, y_test, rf, w_hat=None, k_max=100, num_processes=1, batch_size=5000, verbose=False):
    """
    Calculate various metrics for a given random forest model on a test dataset.

    Parameters:
    - X_test (array-like): The test dataset.
    - y_train (array-like): The training labels.
    - y_test (array-like): The test labels.
    - rf (RandomForest): The random forest model.
    - w_hat (array-like, optional): The weights for weighted prediction. Default is None.
    - k_max (int, optional): The maximum value of k for top-k metrics. Default is 100.
    - num_processes (int, optional): The number of processes for parallel computation. Default is 1.
    - batch_size (int, optional): The batch size for prediction. Default is 5000.
    - verbose (bool, optional): Whether to print progress information. Default is False.

    Returns:
    - results (dict): Dictionary containing various metrics calculated during the top-k loop.
        - 'r2' (array-like): R^2 values.
        - 'corr' (array-like): Correlation coefficients (between forecast and true values).
        - 'cov' (array-like): Covariance values (between forecast and true values).
        - 'se' (array-like): SE values.
        - 'ae' (array-like): AE values.
        - 'crps' (array-like): CRPS values.
        - 'var' (array-like): Variance (of forecast) values.
        - 'bias' (array-like): Bias (of forecast) values.
        - 'pred' (array-like): Predicted values.
        - 'topk_sums' (array-like): Sum of remaining weights for top k values before normalization.
    """

    all_r2 = []
    all_corr = []
    all_cov = []
    all_se = []
    all_ae = []
    all_crps = []
    all_var = []
    all_bias = []
    all_w_topk_sums = []
    all_pred = []

    y_test_mean = np.mean(y_test)

    if w_hat is None:
        y_hat, w_hat = rf.weight_predict(X_test, batch_size=batch_size, return_weights=True)
    else:
        y_hat = rf.predict(X_test)

    y_hat_med = rf.quantile_predict(q=.5, w_all=w_hat, batch_size=batch_size)

    if verbose:
        print("Done Training. Calculating Metrics")

    r2_full_test = calc_r2(y_test, y_hat, y_train=y_train)
    se_full_test = se(y_test, y_hat)
    ae_full_test = ae(y_test, y_hat_med)
    crps_full_test = crps_sample(y_test, y_train, w_hat, return_mean=False)
    cov_full_test = np.cov(y_test, y_hat)[0, 1]
    corr_full_test = np.corrcoef(y_test, y_hat)[0, 1]

    t0 = time.time() if verbose else None

    pool = multiprocessing.Pool(processes=num_processes)
    results = pool.starmap(calc_metrics_topk, [(i, w_hat, y_train, y_test, verbose) for i in range(1, k_max + 1)])
    pool.close()
    pool.join()

    t1 = time.time() if verbose else None

    if verbose:
        print(f"Time taken: {t1 - t0:.2f}s")

    for result in results:
        r2k, se_test, ae_test, crps_test, corr_test, cov_test, var_y_tk, bias_y_tk, w_topk_sums, y_tk = result
        all_r2.append(r2k)
        all_se.append(se_test)
        all_ae.append(ae_test)
        all_crps.append(crps_test)
        all_corr.append(corr_test)
        all_cov.append(cov_test)
        all_var.append(var_y_tk)
        all_bias.append(bias_y_tk)
        all_w_topk_sums.append(w_topk_sums)
        all_pred.append(y_tk)

    all_r2.append(r2_full_test)
    all_se.append(se_full_test)
    all_ae.append(ae_full_test)
    all_crps.append(crps_full_test)
    all_corr.append(corr_full_test)
    all_cov.append(cov_full_test)
    all_var.append(np.var(y_hat))
    all_bias.append(np.mean(y_hat) - y_test_mean)

    all_r2 = np.array(all_r2)
    all_corr = np.array(all_corr)
    all_cov = np.array(all_cov)
    all_se = np.array(all_se)
    all_ae = np.array(all_ae)
    all_crps = np.array(all_crps)
    all_var = np.array(all_var)
    all_bias = np.array(all_bias)
    all_w_topk_sums = np.array(all_w_topk_sums)

    results = {
        'r2': all_r2,
        'corr': all_corr,
        'cov': all_cov,
        'se': all_se,
        'ae': all_ae,
        'crps': all_crps,
        'var': all_var,
        'bias': all_bias,
        'pred': all_pred,
        'topk_sums': all_w_topk_sums
    }

    return results
