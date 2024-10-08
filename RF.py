import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import _forest as forest_utils
from tqdm import tqdm
# from xgboost import XGBRegressor
# import xgboost as xgb
from scipy import stats
import dask.array as da

from utils.score_utils import *
from utils.sparse_utils import *

from numba import jit, njit, prange

import os
import h5py

from scipy.sparse import lil_matrix, csr_matrix
from multiprocessing import Pool

from joblib import Parallel, delayed
import scipy.sparse as scs


class RandomForestWeight:
    """
    Topk Random Forest model class based on the scikit-learn RandomForestRegressor model.
    - hyperparams (dict): Dictionary of hyperparameters for the random forest model.
    - name (str, optional): Name of the random forest model. Default is None.
    Attributes:
    - hyperparams (dict): Dictionary of hyperparameters for the random forest model.
    - r2 (dict): Dictionary to store R-squared values for training and test data.
    - corr (dict): Dictionary to store correlation values for training and test data.
    - mse (dict): Dictionary to store mean squared error values for training and test data.
    - mae (dict): Dictionary to store mean absolute error values for training and test data.
    - name (str): Name of the random forest model.
    - rf (RandomForestRegressor): Random forest regressor scikit-learn model.
    
    """

    def __init__(self, hyperparams, name=None) -> None:
        self.hyperparams = hyperparams

        # make sure those are always set to proper value
        self.hyperparams['n_jobs'] = -1

        self.r2 = {}
        self.corr = {}
        self.mse = {}
        self.mae = {}

        if name is not None:
            self.name = name

        self.rf = RandomForestRegressor(**self.hyperparams)

    def fit(self, X_train, y_train):
        """
        Fits the random forest model using the sklearn API to the training data.

        Parameters:
        - X_train: The input features of the training data.
        - y_train: The target values of the training data.
        """

        self.X_train = X_train
        self.y_train = y_train

        self.rf.fit(self.X_train, self.y_train)

        self.n_trees = len(self.rf.estimators_)

    def predict(self, X):
        """
        Predicts the target variable using the sklearn API for the given input data.

        Parameters:
            X (array-like): Input data of shape (n_samples, n_features).

        Returns:
            array-like: Predicted target variable of shape (n_samples,).
        """
        return self.rf.predict(X)

    def weight_predict(self, X, top_k=None, w_th=None, return_weights=True, batch_size=5000, verbose=True):
        """
        Predicts the mean for the given input data by using weights.

        Parameters:
        - X: Input data to be predicted.
        - top_k: Number of top weights to consider. If specified, only the top_k weights will be used for prediction.
        - w_th: Weight threshold. If specified, weights below this threshold will be set to 0.
        - return_weights: Flag indicating whether to return the weights along with the predicted output.
        - batch_size: Batch size for processing the input data.
        - verbose: Flag indicating whether to print verbose output.

        Returns:
        - If return_weights is True, returns a tuple containing the predicted output and the weights.
        - If return_weights is False, returns the predicted output.
        """
        if top_k and w_th:
            print("Can't give both top_k and w_th!")
            return -1

        w_all = self.get_rf_weights2(X)  #, verbose=verbose)

        if w_th is not None:

            w_all = np.where(w_all < w_th, 0., w_all)
            w_all = w_all / w_all.sum(axis=1)[:, None]

        elif top_k is not None:
            top_k_row_idx = np.argpartition(w_all, -top_k, axis=1)[:, -top_k:]
            i = np.indices(top_k_row_idx.shape)[0]

            w_all_topk = np.zeros(w_all.shape)
            w_all_topk[i, top_k_row_idx] = w_all[i, top_k_row_idx]
            w_all = w_all_topk / w_all_topk.sum(axis=1)[:, None]

        else:
            y_pred = w_all @ self.y_train

        if return_weights:
            return y_pred, w_all

        return y_pred

    def evaluate(self, X_test, y_test, verbose=True):
        """
        Evaluates the random forest model on the test data.
        Stores the evaluation metrics in the class attributes.
        Parameters:
        - X_test: The input features of the test data.
        - y_test: The target values of the test data.
        - verbose: Flag indicating whether to print verbose output.
        
        Returns:
        - None
        """

        self.r2['train'] = self.rf.score(self.X_train, self.y_train)
        self.r2['test'] = self.rf.score(X_test, y_test)

        y_hat = self.predict(X_test)
        y_hat_train = self.predict(self.X_train)

        self.r2['test_tr'] = calc_r2(y_test, y_hat, y_train=self.y_train)

        if verbose is True:
            print(f"RF Sklearn-R2 Train: {self.r2['train']:.2f}")
            print(f"RF Sklearn-R2 Test: {self.r2['test']:.2f}")
            print(f"RF Train-R2 Test: {self.r2['test_tr']:.2f}")

        self.corr['train'] = np.corrcoef(self.y_train, y_hat_train)[0, 1]
        self.corr['test'] = np.corrcoef(y_test, y_hat)[0, 1]

        if verbose is True:
            print(f"RF Correlation Train: {self.corr['train']:.2f}")
            print(f"RF Correlation Test: {self.corr['test']:.2f}")

        self.mse['train'] = mse(self.y_train, y_hat_train)
        self.mse['test'] = mse(y_test, y_hat)

        if verbose is True:
            print(f"RF MSE Train: {self.mse['train']:.2f}")
            print(f"RF MSE Test: {self.mse['test']:.2f}")

        self.mae['train'] = mae(self.y_train, self.quantile_predict(q=.5, X_test=self.X_train))
        self.mae['test'] = mae(y_test, self.quantile_predict(q=.5, X_test=X_test))

        if verbose is True:
            print(f"RF MAE Train: {self.mae['train']:.2f}")
            print(f"RF MAE Test: {self.mae['test']:.2f}")

        self.se = se(y_test, y_hat)

    def get_inbag_idx(self):
        """
        Returns the sampled indices of the training data for each tree in the RF.

        Returns:
            tuple: A tuple containing two arrays:
                - The first array contains the sampled indices for each tree in the RF.
                - The second array contains the count of sampled indices for each tree in the RF.
        """

        n_samples = len(self.X_train)

        n_samples_bootstrap = forest_utils._get_n_samples_bootstrap(n_samples, self.rf.max_samples)

        sampled_indices_trees = []
        sampled_indices_trees_count = []

        for estimator in self.rf.estimators_:

            sampled_indices = forest_utils._generate_sample_indices(estimator.random_state, n_samples,
                                                                    n_samples_bootstrap)
            sampled_indices_trees.append(sampled_indices)

            sampled_indices_hist = np.bincount(sampled_indices, minlength=n_samples_bootstrap)
            sampled_indices_trees_count.append(sampled_indices_hist)

        return np.vstack(sampled_indices_trees), np.vstack(sampled_indices_trees_count)

    @staticmethod
    @njit(parallel=True)
    def weights_loop(n_trees, len_train, len_test, pred1, pred2, inbag=np.array([[]])):
        """
        Calculate the weights for each test observation based on the random forest predictions in parallel.
        Parameters:
        - n_trees (int): The number of trees in the random forest.
        - len_train (int): The length of the training data.
        - len_test (int): The length of the test data.
        - pred1 (numpy.ndarray): The predictions of the training data by the random forest.
        - pred2 (numpy.ndarray): The predictions of the test data by the random forest.
        - inbag (numpy.ndarray, optional): The bootstrapped training samples for each tree. Default is an empty array.
        Returns:
        - w_all (numpy.ndarray): The weights for each test observation.
        """

        w_all = np.zeros((len_test, len_train), dtype=np.float32)

        for i in prange(len_test):
            aux1 = pred2[i]
            # aux2 = np.zeros((n, n_trees))

            for j in prange(n_trees):
                # for tree j: which training obs. (pred1) are in the same leaf as test obs i (aux1)
                which_same = np.where(pred1[:, j] == aux1[j])[0]
                # for tree j: how often were these training obs. present in bootstrapped training sample?
                # should be > 0 for all elements.
                how_often = inbag[j][which_same]
                # how_often = inbag[j,which_same]
                ho_sum = sum(how_often)

                # aux2[which_same, j] = aux2[which_same, j] + how_often

                w_all[i][which_same] = w_all[i][which_same] + ((how_often / ho_sum) / n_trees)
                # w_all[i, which_same] = w_all[i, which_same] + ((how_often/ho_sum) / n_trees)

        return w_all

    def get_rf_weights2(self, X_test):  #, return_checks=False, verbose=True):
        """
        Calculate the weights for random forest predictions.
        Parameters:
        - X_test: numpy array
            The test data for which weights need to be calculated.
        Returns:
        - w_all: numpy array
            The calculated weights for the random forest predictions.
        """
        _, inbag = self.get_inbag_idx()

        inbag = inbag.astype(np.int16)

        if inbag.max() < np.iinfo(np.int16).max:
            inbag = inbag.astype(np.int16)

        # leaf indices for each tree for each training observation
        pred1 = self.rf.apply(self.X_train)

        # leaf indices for each tree for each test observation
        pred2 = self.rf.apply(X_test)

        pred12_max = max((pred1.max(), pred2.max()))

        if pred12_max < np.iinfo(np.int16).max:
            pred1 = pred1.astype(np.int16)
            pred2 = pred2.astype(np.int16)
        elif pred12_max < np.iinfo(np.int32).max:
            pred1 = pred1.astype(np.int32)
            pred2 = pred2.astype(np.int32)

        if max((pred1.max(), pred2.max())) < np.iinfo(np.int32).max:
            pred1 = pred1.astype(np.int32)
            pred2 = pred2.astype(np.int32)

        w_all = self.weights_loop(n_trees=self.n_trees,
                                  len_train=len(self.X_train),
                                  len_test=len(X_test),
                                  pred1=pred1,
                                  pred2=pred2,
                                  inbag=inbag)

        return w_all

    @staticmethod
    def process_row(n_trees, pred1, pred2, inbag, w_all, idx_sort=None):
        """
        Process row of the random forest sparse predictions.

        Args:
            n_trees (int): The number of trees in the random forest.
            pred1 (numpy.ndarray): The predictions from the first set of trees.
            pred2 (numpy.ndarray): The predictions from the second set of trees.
            inbag (numpy.ndarray): The inbag matrix indicating which samples were used to train each tree.
            w_all (scipy.sparse.lil_matrix): Empty lil_matrix to be filled for a single row.
            idx_sort (numpy.ndarray, optional): The indices to sort the weight matrix. Defaults to None.

        Returns:
            scipy.sparse.lil_matrix: The calculated weight row for a test case.

        """

        for j in range(n_trees):
            which_same = np.where(pred1[:, j] == pred2[j])[0]
            how_often = inbag[j][which_same]
            ho_sum = how_often.sum()
            divider = ho_sum * n_trees
            w_all[0, which_same] = w_all[0, which_same] + (how_often / divider)

        if idx_sort is not None:
            return w_all[:, idx_sort]
        return w_all

    def get_rf_weights_sparse(self, X_test, sort=True):
        """
        Calculates the sparse random forest weights for the given test data in parallel.

        Parameters:
        - X_test: array-like
            The test data for which to calculate the weights.
        - sort: bool, optional
            Whether to sort the results by the target variable. Default is True.

        Returns:
        - w_all: scipy.sparse.csr_matrix
            The sparse matrix containing the calculated weights.
        """
        _, inbag = self.get_inbag_idx()

        del _

        if inbag.max() < np.iinfo(np.int16).max:
            inbag = inbag.astype(np.int16)

        # leaf indices for each tree for each training observation
        pred1 = self.rf.apply(self.X_train)

        # leaf indices for each tree for each test observation
        pred2 = self.rf.apply(X_test)

        len_X_test = len(X_test)

        del X_test

        if max((pred1.max(), pred2.max())) < np.iinfo(np.int32).max:
            pred1 = pred1.astype(np.int32)
            pred2 = pred2.astype(np.int32)

        print('Done with inbag, pred1, pred2')

        w_all = lil_matrix((1, len(self.X_train)), dtype=np.float32)
        if sort:
            idx_sort = np.argsort(self.y_train)
            results = Parallel(n_jobs=-1)(
                delayed(self.process_row)(self.n_trees, pred1, pred2[i], inbag, w_all, idx_sort)
                for i in range(len_X_test))
        else:
            results = Parallel(n_jobs=-1)(
                delayed(self.process_row)(self.n_trees, pred1, pred2[i], inbag, w_all) for i in range(len(X_test)))

        w_all = scs.vstack(results)
        return w_all.tocsr()

    def get_rf_weights(self, X_test, return_checks=False, verbose=True):
        """
        [This method should not be used anymore unless for testing/understanding purposes. Use ger_rf_weights2 instead.]
        
        Calculate the weights for random forest predictions.

        Parameters:
        - X_test: DataFrame or array-like
            The test data for which to calculate the weights.
        - return_checks: bool, optional
            Whether to return additional checks along with the weights. Default is False.
        - verbose: bool, optional
            Whether to display progress bar during calculation. Default is True.

        Returns:
        - w_all: ndarray
            The calculated weights for the test data.
        - pred3: ndarray, optional
            The predictions for the test data from each tree in the random forest. Only returned if `return_checks` is True.
        - pred3_check: ndarray, optional
            Additional checks for the predictions. Only returned if `return_checks` is True.
        """
        inbag_idx, inbag = self.get_inbag_idx()

        pred1 = self.rf.apply(self.X_train)
        pred2 = self.rf.apply(X_test)
        pred3 = np.zeros((len(X_test), self.n_trees))

        for i, tree in enumerate(self.rf.estimators_):
            pred3[:, i] = tree.predict(X_test.values)

        w_all = np.zeros((len(X_test), len(self.X_train)))

        pred3_check = np.empty((len(w_all), self.n_trees))
        pred3_check[:] = np.nan

        # aux2 = np.zeros((n, n_trees))
        for i in tqdm(range(len(X_test)), disable=not verbose):
            aux1 = pred2[i]
            # aux2 = np.zeros((n, n_trees))

            for j in range(self.n_trees):
                # for tree j: which training obs. (pred1) are in the same leaf as test obs i (aux1)
                which_same = np.where(pred1[:, j] == aux1[j])[0]
                # for tree j: how often were these training obs. present in bootstrapped training sample?
                # should be > 0 for all elements.
                how_often = inbag[j, which_same]
                ho_sum = sum(how_often)

                # aux2[which_same, j] = aux2[which_same, j] + how_often

                w_all[i, which_same] = w_all[i, which_same] + ((how_often / ho_sum) / self.n_trees)

                pred3_check[i, j] = (self.y_train[which_same] * how_often).sum() / ho_sum

        if not np.equal(pred3, pred3_check).all():
            print("WARNING! Some predictions are off!")

        if not np.allclose(w_all.sum(axis=1), np.ones(len(w_all)), rtol=1e-10):
            print("WARNING! Some weights are != 1")

        pred3_mean = pred3.mean(axis=1)
        pred3_check_mean = pred3_check.mean(axis=1)

        if not np.equal(pred3_mean, pred3_check_mean).all():
            print("WARNING! Some mean predictions are off")

        if not np.corrcoef(pred3_mean, pred3_check_mean)[0][1] >= .999999:
            print("WARNING! Mean predictions are NOT perfectly correlated")

        pred3_weight_pred = w_all @ self.y_train
        if not np.allclose(pred3_mean, pred3_weight_pred, rtol=1e-10):
            print("WARNING! Smoother predictions don't match perfectly")

        self.w_all = w_all

        if return_checks is True:
            return w_all, pred3, pred3_check
        else:
            return w_all

    def sparse_quantile_predict(self, q, w_all):
        """
        Predicts quantiles for given values of q for a sparse w_all.

        Parameters:
        - q: A scalar or an array-like object containing quantile values.
        - w_all: An array-like object containing weights.

        Returns:
        - result: A numpy array of shape (len(ecdfs), len(q)) if q is an array-like object, else a numpy array of shape (len(ecdfs),) if q is a scalar.
        """

        idx_sort = np.argsort(self.y_train)
        y_sort = self.y_train[idx_sort]

        ecdfs = sparse_cumsum(w_all)

        if isinstance(q, (list, pd.core.series.Series, np.ndarray)):

            result = np.zeros((len(ecdfs), len(q)))

            for i, q_i in enumerate(q):

                q_idxs = np.asarray((ecdfs >= q_i).argmax(axis=1)).squeeze()

                result[:, i] = (y_sort[q_idxs - 1] + y_sort[q_idxs]) * .5

            return result

        else:

            q_idxs = np.asarray((ecdfs >= q).argmax(axis=1)).squeeze()

            return (y_sort[q_idxs - 1] + y_sort[q_idxs]) * .5

    def quantile_predict(self, q, X_test=None, w_all=None, top_k=None, batch_size=5000, verbose=False):
        """
        Calculate quantile predictions for given data X_test and quantile level(s) q.

        Parameters:
        - q (float or list-like): The quantile(s) to calculate.
        - X_test (array-like, optional): The test data. Default is None.
        - w_all (array-like, optional): The weight matrix. Default is None.
        - top_k (int, optional): The number of top k rows to consider. Default is None.
        - batch_size (int, optional): The batch size for calculating quantiles in batches. Default is 5000.
        - verbose (bool, optional): Whether to print verbose output. Default is False.

        Returns:
        - array-like: The quantile predictions.
        """

        if X_test is None and w_all is not None:
            if verbose is True:
                print("Will calculate quantiles with given weight matrix")
        elif X_test is not None and w_all is None:
            w_all = self.get_rf_weights2(X_test)
            if top_k is not None:
                top_k_row_idx = np.argpartition(w_all, -top_k, axis=1)[:, -top_k:]
                i = np.indices(top_k_row_idx.shape)[0]

                w_all_topk = np.zeros(w_all.shape)
                w_all_topk[i, top_k_row_idx] = w_all[i, top_k_row_idx]
                w_all = w_all_topk / w_all_topk.sum(axis=1)[:, None]
        elif X_test is None and w_all is None:
            print("X_test and w_all cannot both be None!")
            return -1

        idx_sort = np.argsort(self.y_train)
        if isinstance(self.y_train, pd.DataFrame) or isinstance(self.y_train, pd.Series):
            y_sort = self.y_train.values[idx_sort]
        else:
            y_sort = self.y_train[idx_sort]

        ecdfs = np.cumsum(w_all[:, idx_sort], axis=1)

        del w_all
        del idx_sort

        if isinstance(q, (list, pd.core.series.Series, np.ndarray)):

            result = np.zeros((len(ecdfs), len(q)))

            for i, q_i in enumerate(q):

                q_idxs = np.argmax(ecdfs >= q_i, axis=1)

                # special case if q is really small and q_idxs may be 0
                if q_idxs == 0:
                    q_idxs = 1
                result[:, i] = (y_sort[q_idxs - 1] + y_sort[q_idxs]) * .5
                # result[:,i] = (y_sort[q_idxs] + y_sort[q_idxs+1]) * .5

            return result

        else:

            q_idxs = np.argmax(ecdfs >= q, axis=1)

            # special case if q is really small and q_idxs may be 0
            if q_idxs == 0:
                q_idxs = 1

            return (y_sort[q_idxs - 1] + y_sort[q_idxs]) * .5
            # indices_tmp = np.where(ecdfs < q, np.arange(ecdfs.shape[1])[None, :], -1)
            # q_idxs = indices_tmp.max(axis=1)

            # return (y_sort[q_idxs] + y_sort[q_idxs+1]) * .5
