#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import dask.array as da
from tqdm import tqdm
from itertools import product
import pickle
import os
from sklearn.model_selection import train_test_split
from scipy.sparse import save_npz, load_npz
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
import matplotlib.patches as mpatches

cwd = os.getcwd()

SEED = 7531
np.random.seed(SEED)

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 12,
    #"pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": True,
    "lines.antialiased": True,
    "patch.antialiased": True,
    'axes.linewidth': 0.1
})

from utils.plotting_helpers import *
from utils.score_utils import *

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

from RF import RandomForestWeight

import openml

#%%
all_ids = list(range(44132, 44149)) + [44026, 44027, 44028] + [45032, 45034]
all_ids.remove(44135)

paper_ids = [ds for ds in all_ids if ds not in (44026, 44027, 44028)]

all_ds = [openml.datasets.get_dataset(pid) for pid in all_ids]
paper_ds = [openml.datasets.get_dataset(pid) for pid in paper_ids]

targets = [ds.default_target_attribute for ds in all_ds]
ds_names = [ds.name for ds in all_ds]
ds_names_paper = [ds.name for ds in paper_ds]

#%%

OVERVIEW = True

if OVERVIEW:
    dataset_lengths = []
    num_regressors = []
    target_names = []

    for ds in paper_ds:
        X, _, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")
        dataset_lengths.append(len(X))
        num_regressors.append(X.shape[1])
        target_names.append(ds.default_target_attribute)

    df_overview = pd.DataFrame({
        'Dataset': ds_names_paper,
        'Length': dataset_lengths,
        'Num. Regressors': num_regressors,
        'Target': target_names,
        'OpenML ID': ['\\url{https://www.openml.org/d/' + str(pid) + "}" for pid in paper_ids]
    }).set_index("Dataset")


#%%
def get_topk_weight(w_full, k=3, return_sum=False):
    row_idx = np.argpartition(w_full, -k, axis=1)[:, -k:]
    i = np.indices(row_idx.shape)[0]

    w_tmp = np.zeros(w_full.shape, dtype=np.float32)

    w_tmp[i, row_idx] = w_full[i, row_idx]

    w_topk_sums = w_tmp.sum(axis=1)
    w_k = w_tmp / w_topk_sums[:, None]

    if return_sum:
        return w_k, w_topk_sums
    return w_k


#%%
import gc

TEST_SIZE = 0.3
VAL_SIZE = 0.25

NO_PROCESSES = 1

N_TREES = 1000

STANDARDIZE_Y = False

K = 3
N_CV = 5

LARGE_DS_NAMES = ['nyc-taxi-green-dec-2016', 'medical_charges', 'delays_zurich_transport']

map_int_hp = {}
map_int_hp_sparse = {}

for i, ds in enumerate(paper_ds):

    file_name = cwd + f"/results/openml/hptuningCV_top{K}_{ds.name.replace(' ', '_')}.pkl"

    if os.path.exists(file_name):
        print("Skipping", ds.name, "as it is already processed")
        continue

    # if ds.name in LARGE_DS_NAMES:
    #     print('Skipping', ds.name, 'due to size')
    #     continue

    print(ds.name, 'target:', ds.default_target_attribute)

    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")

    if STANDARDIZE_Y:
        y_mean = y.mean()
        y_std = y.std()
        y = (y - y_mean) / y_std

    if ds.name == 'sulfur':
        X = X.drop(columns=['y2'])

    # if len(X) < 100_000:
    #     print('Skipping', ds.name, 'due to size')
    #     continue

    if ds.name == 'delays_zurich_transport':
        with open(cwd + f'/data/delays_zurich_transport_randixs02.pkl', 'rb') as file:
            rand_idxs = pickle.load(file)

        X = X.iloc[rand_idxs]
        y = y.iloc[rand_idxs]

    # df_train, df_test, y_train, y_test = train_test_split(X, y.values, test_size=TEST_SIZE, random_state=SEED)
    df_train, _, y_train, _ = train_test_split(X, y.values, test_size=TEST_SIZE, random_state=SEED)

    X_train = df_train.values
    # X_test = df_test.values

    del X, y
    del df_train, _  #df_test

    print(f'Dataset split. X_train shape: {X_train.shape}')  #, X_test shape: {X_test.shape}')

    if ds.name in LARGE_DS_NAMES:

        # if ds.name != 'medical_charges':
        #     continue

        gc.collect()

        search_grid = {
            'max_depth': [None],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10, 15, 20, 50],
            'min_samples_split': [5],  #[2, 5, 8, 11, 14, 17],
            'max_features': [0.333, 0.5, 1.0]
        }

        if ds.name == 'delays_zurich_transport':
            if os.path.exists(cwd + f'/data/delays_zurich_transport_randixs015.pkl'):
                with open(cwd + f'/data/delays_zurich_transport_hptuning_randixs015.pkl', 'rb') as file:
                    rand_idxs = pickle.load(file)
            else:
                rand_idxs = np.random.choice(len(X_train), size=int(len(X_train) * 0.15), replace=False)
                with open(cwd + f'/data/delays_zurich_transport_hptuning_randixs015.pkl', 'wb') as file:
                    pickle.dump(rand_idxs, file)

        elif ds.name == 'nyc-taxi-green-dec-2016':
            if os.path.exists(cwd + f'/data/nyc_taxi_randixs03.pkl'):
                with open(cwd + f'/data/nyc_taxi_hptuning_randixs03.pkl', 'rb') as file:
                    rand_idxs = pickle.load(file)
            else:
                rand_idxs = np.random.choice(len(X_train), size=int(len(X_train) * 0.3), replace=False)
                with open(cwd + f'/data/nyc_taxi_hptuning_randixs03.pkl', 'wb') as file:
                    pickle.dump(rand_idxs, file)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train[rand_idxs] if ds.name != 'medical_charges' else X_train,
            y_train[rand_idxs] if ds.name != 'medical_charges' else y_train,
            test_size=VAL_SIZE,
            random_state=SEED)

        run_counter = 0
        all_results = {}
        for md in search_grid['max_depth']:
            for mss in search_grid['min_samples_split']:
                for msl in search_grid['min_samples_leaf']:
                    for mf in search_grid['max_features']:

                        if mf == 0.5 and ds.name == 'medical_charges':
                            run_counter += 1
                            continue

                        result = {}

                        # if run_counter % 10 == 0:
                        print(f"Run No. {run_counter}")

                        run_counter += 1

                        hyperparams = dict(n_estimators=N_TREES,
                                           random_state=SEED,
                                           n_jobs=-1,
                                           max_features=mf,
                                           min_samples_split=mss,
                                           min_samples_leaf=msl)

                        map_int_hp_sparse[run_counter] = hyperparams

                        se_full = []
                        se_k = []

                        crps_full = []
                        crps_k = []

                        rf = RandomForestWeight(hyperparams=hyperparams, name='rf_' + ds.name)
                        rf.fit(X_train, y_train)

                        # if ds.name == 'medical_charges':
                        y_hat, w_all = rf.weight_predict(X_val, return_weights=True)

                        # w_all = w_all.tocsr
                        w_k = get_topk_weight(w_all, k=K, return_sum=False)
                        # top_idx, top_dataidx = top_n_idx_sparse(w_all, K)
                        # w_k = sparsify_csr(w_all, top_idx, top_dataidx, return_sum=False)
                        y_hat_k = y_train @ w_k.T

                        mse_full_cv = se(y_val, y_hat).mean()
                        mse_k_cv = se(y_val, y_hat_k).mean()

                        crps_full_cv = crps_sample(y_val, y_train, w_all, return_mean=True)
                        crps_k_cv = crps_sample(y_val, y_train, w_k, return_mean=True)

                        se_full.append(mse_full_cv)
                        se_k.append(mse_k_cv)
                        crps_full.append(crps_full_cv)
                        crps_k.append(crps_k_cv)
                        # else:
                        #     gc.collect()

                        #     hp_string = f"md_{md}_mss_{mss}_msl_{msl}_mf_{mf}"
                        #     weight_path = cwd + '/weight_storage/rf_' + ds.name + '_weights_' + hp_string + '.npz'

                        #     if os.path.exists(weight_path):
                        #         w_all_sparse = load_npz(weight_path).tocsr()
                        #         print("Weights loaded...")
                        #     else:
                        #         w_all_sparse = rf.get_rf_weights_sparse(X_val, sort=True)
                        #         save_npz(weight_path, w_all_sparse)
                        #         print("Weights calculated and stored...")

                        #     gc.collect()

                        #     idx_sort = np.argsort(y_train)
                        #     y_train = y_train[idx_sort]

                        #     y_hat = rf.predict(X_val)

                        #     mse_full_cv = se(y_val, y_hat).mean()
                        #     crps_full_cv = crps_sample_sparse2(y_val, y_train, w_all_sparse, dat_ordered=True).mean()

                        #     del y_hat

                        #     top_idx, top_dataidx = top_n_idx_sparse(w_all_sparse, K)
                        #     w_k = sparsify_csr(w_all_sparse, top_idx, top_dataidx, return_sum=False)

                        #     y_hat_k = w_k @ y_train

                        #     mse_k_cv = se(y_val, y_hat_k).mean()
                        #     crps_k_cv = crps_sample_sparse2(y_val, y_train, w_k, dat_ordered=True).astype(np.float32)

                        #     se_full.append(mse_full_cv)
                        #     se_k.append(mse_k_cv)
                        #     crps_full.append(crps_full_cv)
                        #     crps_k.append(crps_k_cv)

                        result['se_full'] = np.mean(se_full)
                        result['se_k'] = np.mean(se_k)
                        result['crps_full'] = np.mean(crps_full)
                        result['crps_k'] = np.mean(crps_k)

                        all_results[run_counter] = result

    else:

        del X, y
        del df_train, df_test

        gc.collect()

        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)

        search_grid = {
            'max_depth': [None],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50],
            'min_samples_split': [5],  #[2, 5, 8, 11, 14, 17],
            'max_features': [0.333, 'sqrt', 0.5, 1.0],
        }

        kf = KFold(
            n_splits=N_CV)  # nor random_state required, as there is not shuffling, random_state=SEED, shuffle=False)

        run_counter = 0
        all_results = {}
        for md in search_grid['max_depth']:
            for mss in search_grid['min_samples_split']:
                for msl in search_grid['min_samples_leaf']:
                    for mf in search_grid['max_features']:

                        result = {}

                        if run_counter % 20 == 0:
                            print(f"Run No. {run_counter}")

                        run_counter += 1

                        hyperparams = dict(n_estimators=N_TREES,
                                           random_state=SEED,
                                           n_jobs=-1,
                                           max_features=mf,
                                           min_samples_split=mss,
                                           min_samples_leaf=msl)

                        map_int_hp[run_counter] = hyperparams

                        se_full = []
                        se_k = []

                        crps_full = []
                        crps_k = []

                        for train_index, test_index in kf.split(X_train):
                            X_train_cv, X_test_cv = X_train[train_index], X_train[test_index]
                            y_train_cv, y_test_cv = y_train[train_index], y_train[test_index]

                            rf = RandomForestWeight(hyperparams=hyperparams, name='rf_' + ds.name)
                            rf.fit(X_train_cv, y_train_cv)

                            y_hat, w_all = rf.weight_predict(X_test_cv, return_weights=True)

                            w_k = get_topk_weight(w_all, k=K, return_sum=False)
                            y_hat_k = y_train_cv @ w_k.T

                            mse_full_cv = se(y_test_cv, y_hat).mean()
                            mse_k_cv = se(y_test_cv, y_hat_k).mean()

                            crps_full_cv = crps_sample(y_test_cv, y_train_cv, w_all, return_mean=True)
                            crps_k_cv = crps_sample(y_test_cv, y_train_cv, w_k, return_mean=True)

                            se_full.append(mse_full_cv)
                            se_k.append(mse_k_cv)
                            crps_full.append(crps_full_cv)
                            crps_k.append(crps_k_cv)

                        result['se_full'] = np.mean(se_full)
                        result['se_k'] = np.mean(se_k)
                        result['crps_full'] = np.mean(crps_full)
                        result['crps_k'] = np.mean(crps_k)

                        all_results[run_counter] = result

    with open(file_name, 'wb') as file:
        pickle.dump(all_results, file)

#%%
search_grid = {
    'max_depth': [None],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50],
    'min_samples_split': [5],  #[2, 5, 8, 11, 14, 17],
    'max_features': [0.333, 'sqrt', 0.5, 1.0],
}

search_grid_sparse = {
    'max_depth': [None],
    'min_samples_leaf': [1, 2, 4, 6, 8, 10, 15, 20, 50],
    'min_samples_split': [5],  #[2, 5, 8, 11, 14, 17],
    'max_features': [0.333, 0.5, 1.0],
}

map_int_sparse_hp = {}

run_counter = 0
for md in search_grid['max_depth']:
    for mss in search_grid['min_samples_split']:
        for msl in search_grid['min_samples_leaf']:
            for mf in search_grid['max_features']:

                run_counter += 1

                hyperparams = dict(n_estimators=N_TREES,
                                   random_state=SEED,
                                   n_jobs=-1,
                                   max_features=mf,
                                   min_samples_split=mss,
                                   min_samples_leaf=msl)

                map_int_hp[run_counter] = hyperparams

run_counter = 0
for md in search_grid_sparse['max_depth']:
    for mss in search_grid_sparse['min_samples_split']:
        for msl in search_grid_sparse['min_samples_leaf']:
            for mf in search_grid_sparse['max_features']:

                run_counter += 1

                hyperparams = dict(n_estimators=N_TREES,
                                   random_state=SEED,
                                   n_jobs=-1,
                                   max_features=mf,
                                   min_samples_split=mss,
                                   min_samples_leaf=msl)

                map_int_sparse_hp[run_counter] = hyperparams

all_results = {}
for ds in ds_names_paper:
    # if ds != 'medical_charges' and ds in LARGE_DS_NAMES:
    #     continue
    file_name = cwd + f"/results/openml/hptuningCV_top{K}_{ds.replace(' ', '_')}.pkl"
    with open(file_name, 'rb') as file:
        all_results[ds] = pickle.load(file)
# %%
best_hps_full_se = {}
best_hps_full_crps = {}

best_hps_k_se = {}
best_hps_k_crps = {}

best_se = {}
best_crps = {}

for ds, results in all_results.items():
    best_se_full = np.inf
    best_crps_full = np.inf
    best_se_k = np.inf
    best_crps_k = np.inf

    hp_mapper = map_int_sparse_hp if ds in LARGE_DS_NAMES else map_int_hp

    for run, res in results.items():
        if res['se_full'] < best_se_full:
            best_se_full = res['se_full']
            best_hps_full_se[ds] = hp_mapper[run]
        if res['crps_full'] < best_crps_full:
            best_crps_full = res['crps_full']
            best_hps_full_crps[ds] = hp_mapper[run]
        if res['se_k'] < best_se_k:
            best_se_k = res['se_k']
            best_hps_k_se[ds] = hp_mapper[run]
        if res['crps_k'] < best_crps_k:
            best_crps_k = res['crps_k']
            best_hps_k_crps[ds] = hp_mapper[run]

        print(f'Best Losses for {ds}:')
        print(f'Full SE: {best_se_full:.4f}, Full CRPS: {best_crps_full:.4f}')
        print(f'K SE: {best_se_k:.4f}, K CRPS: {best_crps_k:.4f}')
        print('-----------------------------------')

    # best_hps_full_se[ds]['se'] = best_se_full
    # best_hps_full_crps[ds]['crps'] = best_crps_full
    # best_hps_k_se[ds]['se'] = best_se_k
    # best_hps_k_crps[ds]['crps'] = best_crps_k

    best_se[ds] = (best_se_full, best_se_k)
    best_crps[ds] = (best_crps_full, best_crps_k)
# %%
relative_se = {}
relative_crps = {}
for k in best_se.keys():
    relative_se[k] = best_se[k][1] / best_se[k][0]
    relative_crps[k] = best_crps[k][1] / best_crps[k][0]

#%%
# relative_se_ttop3 = {}
# relative_crps_ttop3 = {}

# for k in best_hps_k_se.keys():
#     relative_se_ttop3[k]

# %%
df_hps_k_se = pd.DataFrame(best_hps_k_se).T
df_hps_k_se['se'] = np.array(list(best_se.values()))[:, 1]

df_hps_k_crps = pd.DataFrame(best_hps_k_crps).T
df_hps_k_crps['crps'] = np.array(list(best_crps.values()))[:, 1]

df_hps_full_se = pd.DataFrame(best_hps_full_se).T
df_hps_full_se['se'] = np.array(list(best_se.values()))[:, 0]

df_hps_full_crps = pd.DataFrame(best_hps_full_crps).T
df_hps_full_crps['crps'] = np.array(list(best_crps.values()))[:, 0]

df_hps_k_se.to_csv(cwd + f'/results/openml/best_hps_k{K}_se.csv')
df_hps_k_crps.to_csv(cwd + f'/results/openml/best_hps_k{K}_crps.csv')
df_hps_full_se.to_csv(cwd + '/results/openml/best_hps_full_se.csv')
df_hps_full_crps.to_csv(cwd + '/results/openml/best_hps_full_crps.csv')

df_hps_k = df_hps_k_se[['max_features', 'min_samples_leaf']].join(df_hps_k_crps[['max_features', 'min_samples_leaf']],
                                                                  how='left',
                                                                  rsuffix='_crps',
                                                                  lsuffix='_se')
# %%
results = []
results_hp = []
df_hp = []
used_ds = []

considered_ks = [3]

HP_TUNED_RESULTS = False
SHALLOW_RESULTS = False
GRID_SEARCH_RESULTS = True

STANDARDIZE_Y_RESULTS = True

for i, ds in enumerate(ds_names_paper):

    if ds in LARGE_DS_NAMES:
        continue

    used_ds.append(ds)

    file_name_hp = cwd + f"/results/openml/results_{ds.replace(' ', '_')}_hptuned_grid.pkl"
    if os.path.exists(file_name_hp):
        with open(file_name_hp, "rb") as file:
            result = pickle.load(file)
            df_hp.append(result["hyperparams"])
            results_hp.append(result)

    file_name = cwd + f"/results/openml/results_{ds.replace(' ', '_')}.pkl"
    if os.path.exists(file_name):

        with open(file_name, "rb") as file:
            result = pickle.load(file)

            print(i, ds)

            results.append(result)

df_hp = pd.DataFrame(df_hp, index=[d for d in ds_names_paper if d not in LARGE_DS_NAMES])

# %%
considered_ks = [3]
loss = 'se'
loss_label = loss.upper() if loss == 'crps' else 'M' + loss.upper()

df_results = {}
for i, ds in enumerate(used_ds):

    # if ds in ['nyc-taxi-green-dec-2016', 'medical_charges', 'delays_zurich_transport']:
    #     continue

    print(i, ds)

    df_results[ds] = {loss_label + ' Full': results[i][loss][-1].mean()}

    for k in considered_ks:
        df_results[ds][f'{loss_label} Top{k}'] = results[i][loss][k - 1].mean()
        df_results[ds][f'{loss_label} Top{k} Skill'] = 1 - results[i][loss][k - 1].mean() / results[i][loss][-1].mean()
        df_results[ds][f'{loss_label} Top{k} Rel'] = (results[i][loss][k - 1].mean() / results[i][loss][-1].mean())
    df_results[ds]['Best k'] = int(np.argmin(results[i][loss].mean(1)) + 1)
    df_results[ds]['Best k'] = '> 200' if df_results[ds]['Best k'] == len(
        results[i][loss]) else df_results[ds]['Best k']
df_results = pd.DataFrame(df_results).T

print(df_results[[col for col in df_results.columns if "Skill" not in col]].to_latex(float_format="%.4f"))

if loss == 'se':
    df_results[f'MSE Top{K} Rel Tuned'] = np.array(list(best_se.values()))[:, 1] / df_results['MSE Full']
elif loss == 'crps':
    df_results[f'CRPS Top{K} Rel Tuned'] = np.array(list(best_crps.values()))[:, 1] / df_results['CRPS Full']
skill_cols = [col for col in df_results.columns if "Rel" in col]
df_results[skill_cols] = df_results[skill_cols].astype(float)

print(df_results[skill_cols].median())
df_results[skill_cols]
#%%
considered_ks = [3]
loss = 'se'
loss_label = loss.upper() if loss == 'crps' else 'M' + loss.upper()

df_results_hp = {}
for i, ds in enumerate(used_ds):

    # if ds in ['nyc-taxi-green-dec-2016', 'medical_charges', 'delays_zurich_transport']:
    #     continue

    print(i, ds)

    df_results_hp[ds] = {loss_label + ' Full': results_hp[i][loss][-1].mean()}

    for k in considered_ks:
        df_results_hp[ds][f'{loss_label} Top{k}'] = results_hp[i][loss][k - 1].mean()
        df_results_hp[ds][
            f'{loss_label} Top{k} Skill'] = 1 - results_hp[i][loss][k - 1].mean() / results_hp[i][loss][-1].mean()
        df_results_hp[ds][f'{loss_label} Top{k} Rel'] = (results_hp[i][loss][k - 1].mean() /
                                                         results_hp[i][loss][-1].mean())
    df_results_hp[ds]['Best k'] = int(np.argmin(results_hp[i][loss].mean(1)) + 1)
    df_results_hp[ds]['Best k'] = '> 200' if df_results_hp[ds]['Best k'] == len(
        results_hp[i][loss]) else df_results_hp[ds]['Best k']
df_results_hp = pd.DataFrame(df_results_hp).T

print(df_results_hp[[col for col in df_results_hp.columns if "Skill" not in col]].to_latex(float_format="%.4f"))

if loss == 'se':
    df_results_hp['MSE Top3 Rel Tuned'] = np.array(list(best_se.values()))[:, 1] / df_results_hp['MSE Full']
elif loss == 'crps':
    df_results_hp['CRPS Top3 Rel Tuned'] = np.array(list(best_crps.values()))[:, 1] / df_results_hp['CRPS Full']
skill_cols = [col for col in df_results_hp.columns if "Rel" in col]
df_results_hp[skill_cols] = df_results_hp[skill_cols].astype(float)

print(df_results_hp[skill_cols].median())
df_results_hp[skill_cols]

#%%

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(fraction=1.5))

ticks = np.arange(len(relative_se))

ax.set_title('Relative MSE Top3/Std respectively best HPs')

ax.bar(ticks, list(relative_se.values()), color=blue, label='Top3/Std respectively tuned')
ax.set_xticks(ticks)
ax.set_xticklabels(list(relative_se.keys()), rotation=45, ha='right')

ax.axhline(1, color='grey', linestyle=':', linewidth=1, zorder=-99)
ax.set_ylabel('Relative MSE')

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(fraction=1.5))

ticks = np.arange(len(used_ds))

ax.set_title('Relative MSE Top3/Std')

ax.bar(ticks - .2, df_results['MSE Top3 Rel'], width=.4, color=blue, label='Standard')
ax.bar(ticks + .2, df_results['MSE Top3 Rel Tuned'], width=.4, color=och, label='Tuned on Top3')

ax.set_xticks(ticks)
ax.set_xticklabels(df_results.index, rotation=45, ha='right')
ax.set_ylabel('Relative MSE')

ax.axhline(1, color='grey', linestyle=':', linewidth=1, zorder=-99)
ax.legend()

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(fraction=1.5))

ticks = np.arange(len(used_ds))

ax.set_title('Relative CRPS Top3/Std')

ax.bar(ticks - .2, df_results['CRPS Top3 Rel'], width=.4, color=blue, label='Standard')
ax.bar(ticks + .2, df_results['CRPS Top3 Rel Tuned'], width=.4, color=och, label='Tuned on Top3')

ax.set_xticks(ticks)
ax.set_xticklabels(df_results.index, rotation=45, ha='right')
ax.set_ylabel('Relative CRPS')

ax.axhline(1, color='grey', linestyle=':', linewidth=1, zorder=-99)
ax.legend()
# %%
