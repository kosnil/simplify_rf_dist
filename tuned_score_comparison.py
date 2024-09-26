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
# Load results and hyperparams
# %%
results = []
results_hp = []
df_hp = []
used_ds = []

considered_ks = [3]

LARGE_DS_NAMES = ['nyc-taxi-green-dec-2016', 'delays_zurich_transport', 'medical_charges']

GRID_SEARCH_RESULTS = True

for i, ds in enumerate(ds_names_paper):

    # if ds in LARGE_DS_NAMES:
    #     continue

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
            results.append(result)

df_hp_full_se = pd.DataFrame(df_hp, index=[d for d in ds_names_paper])  # if d not in LARGE_DS_NAMES])
df_hp_full_se['random_state'] = SEED

df_hp_top3_se = pd.read_csv(cwd + f'/results/openml/best_hps_k3_se.csv', index_col='Unnamed: 0')
df_hp_top3_se['random_state'] = SEED
df_hp_top3_se['max_depth'] = None
df_hp_top3_se.drop(columns=['se'], inplace=True)

df_hp_full_se['min_samples_split'] = df_hp_full_se['min_samples_split'].astype(int)
df_hp_full_se['min_samples_leaf'] = df_hp_full_se['min_samples_leaf'].astype(int)
df_hp_full_se['n_estimators'] = df_hp_full_se['n_estimators'].astype(int)

df_hp_top3_se['min_samples_split'] = df_hp_top3_se['min_samples_split'].astype(int)
df_hp_top3_se['min_samples_leaf'] = df_hp_top3_se['min_samples_leaf'].astype(int)
df_hp_top3_se['n_estimators'] = df_hp_top3_se['n_estimators'].astype(int)

df_hp_full_crps = pd.read_csv(cwd + f'/results/openml/best_hps_full_crps.csv', index_col='Unnamed: 0')
df_hp_full_crps['random_state'] = SEED
df_hp_full_crps['max_depth'] = None
df_hp_full_crps.drop(columns=['crps'], inplace=True)

df_hp_top3_crps = pd.read_csv(cwd + f'/results/openml/best_hps_k3_crps.csv', index_col='Unnamed: 0')
df_hp_top3_crps['random_state'] = SEED
df_hp_top3_crps['max_depth'] = None
df_hp_top3_crps.drop(columns=['crps'], inplace=True)

df_hp_full_crps['min_samples_split'] = df_hp_full_crps['min_samples_split'].astype(int)
df_hp_full_crps['min_samples_leaf'] = df_hp_full_crps['min_samples_leaf'].astype(int)
df_hp_full_crps['n_estimators'] = df_hp_full_crps['n_estimators'].astype(int)

df_hp_top3_crps['min_samples_split'] = df_hp_top3_crps['min_samples_split'].astype(int)
df_hp_top3_crps['min_samples_leaf'] = df_hp_top3_crps['min_samples_leaf'].astype(int)
df_hp_top3_crps['n_estimators'] = df_hp_top3_crps['n_estimators'].astype(int)


#%%
def convert_hp_types(hp_dict):
    for k, v in hp_dict.items():
        if k == 'max_features':
            try:
                hp_dict[k] = float(v)
            except ValueError:
                hp_dict[k] = v

    return hp_dict


import gc

TEST_SIZE = 0.3
NO_PROCESSES = 1

all_se_full = []
all_se_top3 = []

all_crps_full = []
all_crps_top3 = []

for i, ds in enumerate(paper_ds):

    if ds.name not in used_ds:
        continue

    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")

    if ds.name == 'sulfur':
        X = X.drop(columns=['y2'])

    if ds.name == 'delays_zurich_transport':
        with open(cwd + f'/data/delays_zurich_transport_randixs02.pkl', 'rb') as file:
            rand_idxs = pickle.load(file)
        X = X.iloc[rand_idxs]
        y = y.iloc[rand_idxs]

    df_train, df_test, y_train, y_test = train_test_split(X, y.values, test_size=TEST_SIZE, random_state=SEED)

    X_train = df_train.values
    X_test = df_test.values

    print(f'Dataset split. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')

    del X, y
    del df_train, df_test

    gc.collect()

    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    hp_full_se = convert_hp_types(df_hp_full_se.loc[ds.name].to_dict())
    hp_top3_se = convert_hp_types(df_hp_top3_se.loc[ds.name].to_dict())

    hp_full_crps = convert_hp_types(df_hp_full_crps.loc[ds.name].to_dict())
    hp_top3_crps = convert_hp_types(df_hp_top3_crps.loc[ds.name].to_dict())

    print(ds.name)

    if ds.name in LARGE_DS_NAMES:
        hp_string_full_se = '_'.join(k + '_' + str(v) for k, v in hp_full_se.items())
        hp_string_top3_se = '_'.join(k + '_' + str(v) for k, v in hp_top3_se.items())
        hp_string_full_crps = '_'.join(k + '_' + str(v) for k, v in hp_full_crps.items())
        hp_string_top3_crps = '_'.join(k + '_' + str(v) for k, v in hp_top3_crps.items())

        weight_path_full_se = cwd + '/weight_storage/hptuning_comparision_rf_' + ds.name + '_weights_' + hp_string_full_se + '.npz'
        weight_path_top3_se = cwd + '/weight_storage/hptuning_comparision_rf_' + ds.name + '_weights_' + hp_string_top3_se + '.npz'
        weight_path_full_crps = cwd + '/weight_storage/hptuning_comparision_rf_' + ds.name + '_weights_' + hp_string_full_crps + '.npz'
        weight_path_top3_crps = cwd + '/weight_storage/hptuning_comparision_rf_' + ds.name + '_weights_' + hp_string_top3_crps + '.npz'

        # Full SE
        rf_full = RandomForestWeight(hyperparams=hp_full_se, name='rf_' + ds.name)
        rf_full.fit(X_train, y_train)

        if os.path.exists(weight_path_full_se):
            w_all_sparse = load_npz(weight_path_full_se).tocsr()
            print("Weights loaded...")
        else:
            w_all_sparse = rf_full.get_rf_weights_sparse(X_test, sort=True)
            save_npz(weight_path_full_se, w_all_sparse)
            print("Weights calculated and stored for full SE...")

        gc.collect()

        y_hat = rf_full.predict(X_test)

        mse_full = se(y_test, y_hat).mean()

        del y_hat

        top_idx, top_dataidx = top_n_idx_sparse(w_all_sparse, 3)
        w_k = sparsify_csr(w_all_sparse, top_idx, top_dataidx, return_sum=False)

        y_hat_k = w_k @ y_train

        mse_k = se(y_test, y_hat_k).mean()

        #Full CRPS
        rf_full = RandomForestWeight(hyperparams=hp_full_crps, name='rf_' + ds.name)
        rf_full.fit(X_train, y_train)

        idx_sort = np.argsort(y_train)
        y_sort = y_train[idx_sort]

        if os.path.exists(weight_path_full_crps):
            w_all_sparse = load_npz(weight_path_full_crps).tocsr()
            print("Weights loaded...")
        else:
            w_all_sparse = rf_full.get_rf_weights_sparse(X_test, sort=True)
            save_npz(weight_path_full_crps, w_all_sparse)
            print("Weights calculated and stored for full CRPS...")

        top_idx, top_dataidx = top_n_idx_sparse(w_all_sparse, 3)
        w_k = sparsify_csr(w_all_sparse, top_idx, top_dataidx, return_sum=False)

        crps_full = crps_sample_sparse2(y_test, y_sort, w_all_sparse, dat_ordered=True).mean()
        crps_k = crps_sample_sparse2(y_test, y_sort, w_k, dat_ordered=True).mean()

        all_se_full.append([mse_full, mse_k])
        all_crps_full.append([crps_full, crps_k])

        del rf_full, w_all_sparse, w_k

        # Top3 SE
        rf_top3 = RandomForestWeight(hyperparams=hp_top3_se, name='rf_' + ds.name)
        rf_top3.fit(X_train, y_train)

        if os.path.exists(weight_path_top3_se):
            w_all_sparse = load_npz(weight_path_top3_se).tocsr()
            print("Weights loaded...")
        else:
            w_all_sparse = rf_top3.get_rf_weights_sparse(X_test, sort=True)
            save_npz(weight_path_top3_se, w_all_sparse)
            print("Weights calculated and stored for Top3 SE...")

        gc.collect()

        y_hat = rf_top3.predict(X_test)

        mse_top3 = se(y_test, y_hat).mean()

        del y_hat

        top_idx, top_dataidx = top_n_idx_sparse(w_all_sparse, 3)
        w_k = sparsify_csr(w_all_sparse, top_idx, top_dataidx, return_sum=False)

        y_hat_k = w_k @ y_train

        mse_k = se(y_test, y_hat_k).mean()

        #Top3 CRPS
        rf_top3 = RandomForestWeight(hyperparams=hp_top3_crps, name='rf_' + ds.name)
        rf_top3.fit(X_train, y_train)

        if os.path.exists(weight_path_top3_crps):
            w_all_sparse = load_npz(weight_path_top3_crps).tocsr()
            print("Weights loaded...")
        else:
            w_all_sparse = rf_top3.get_rf_weights_sparse(X_test, sort=True)
            save_npz(weight_path_top3_crps, w_all_sparse)
            print("Weights calculated and stored for Top3 CRPS...")

        top_idx, top_dataidx = top_n_idx_sparse(w_all_sparse, 3)
        w_k = sparsify_csr(w_all_sparse, top_idx, top_dataidx, return_sum=False)

        crps_top3 = crps_sample_sparse2(y_test, y_sort, w_all_sparse, dat_ordered=True).mean()
        crps_k = crps_sample_sparse2(y_test, y_sort, w_k, dat_ordered=True).mean()

        all_se_top3.append([mse_top3, mse_k])
        all_crps_top3.append([crps_top3, crps_k])

        del rf_top3, w_all_sparse, w_k

    else:
        rf_full = RandomForestWeight(hyperparams=hp_full_se, name='rf_' + ds.name)
        rf_full.fit(X_train, y_train)

        y_hat, w_all = rf_full.weight_predict(X_test, return_weights=True)

        w_k = get_topk_weight(w_all, k=3, return_sum=False)
        y_hat_k = y_train @ w_k.T

        mse_full = se(y_test, y_hat).mean()
        mse_k = se(y_test, y_hat_k).mean()

        rf_full = RandomForestWeight(hyperparams=hp_full_crps, name='rf_' + ds.name)
        rf_full.fit(X_train, y_train)

        y_hat, w_all = rf_full.weight_predict(X_test, return_weights=True)

        w_k = get_topk_weight(w_all, k=3, return_sum=False)
        y_hat_k = y_train @ w_k.T

        crps_full = crps_sample(y_test, y_train, w_all, return_mean=True)
        crps_k = crps_sample(y_test, y_train, w_k, return_mean=True)

        all_se_full.append([mse_full, mse_k])
        all_crps_full.append([crps_full, crps_k])

        del rf_full, w_all, w_k

        rf_top3 = RandomForestWeight(hyperparams=hp_top3_se, name='rf_' + ds.name)
        rf_top3.fit(X_train, y_train)

        y_hat, w_all = rf_top3.weight_predict(X_test, return_weights=True)

        w_k = get_topk_weight(w_all, k=3, return_sum=False)
        y_hat_k = y_train @ w_k.T

        mse_full = se(y_test, y_hat).mean()
        mse_k = se(y_test, y_hat_k).mean()

        rf_top3 = RandomForestWeight(hyperparams=hp_top3_crps, name='rf_' + ds.name)
        rf_top3.fit(X_train, y_train)

        _, w_all = rf_top3.weight_predict(X_test, return_weights=True)

        w_k = get_topk_weight(w_all, k=3, return_sum=False)

        crps_full = crps_sample(y_test, y_train, w_all, return_mean=True)
        crps_k = crps_sample(y_test, y_train, w_k, return_mean=True)

        all_se_top3.append([mse_full, mse_k])
        all_crps_top3.append([crps_full, crps_k])

all_se_top3 = np.array(all_se_top3)
all_se_full = np.array(all_se_full)

all_crps_full = np.array(all_crps_full)
all_crps_top3 = np.array(all_crps_top3)
#%%
considered_ks = [3]
loss = 'se'
loss_label = loss.upper() if loss == 'crps' else 'M' + loss.upper()

df_results = {}
for i, ds in enumerate(used_ds):
    df_results[ds] = {loss_label + ' Full': results[i][loss][-1].mean()}

    for k in considered_ks:
        df_results[ds][f'{loss_label} Top{k}'] = results[i][loss][k - 1].mean()
        df_results[ds][f'{loss_label} Top{k} Skill'] = 1 - results[i][loss][k - 1].mean() / results[i][loss][-1].mean()
        df_results[ds][f'{loss_label} Top{k} Rel'] = (results[i][loss][k - 1].mean() / results[i][loss][-1].mean())
    df_results[ds]['Best k'] = int(np.argmin(results[i][loss].mean(1)) + 1)
    df_results[ds]['Best k'] = '> 200' if df_results[ds]['Best k'] == len(
        results[i][loss]) else df_results[ds]['Best k']
df_results = pd.DataFrame(df_results).T

skill_cols = [col for col in df_results.columns if "Rel" in col]
df_results[skill_cols] = df_results[skill_cols].astype(float)

print(df_results[skill_cols])

df_results[skill_cols]

skill_cols = [f'{loss_label} Full'] + skill_cols  #+ ['Best k']

if loss == 'crps':
    crps_standard = df_results['CRPS Top3 Rel'].values
elif loss == 'se':
    se_standard = df_results['MSE Top3 Rel'].values

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(fraction=1.5))

ticks = np.arange(len(used_ds))

ax.bar(ticks - 0.2, all_se_full[:, 1] / all_se_full[:, 0], width=0.4, color=blue, label='Top3/Full tuned on Full')
ax.bar(ticks + 0.2, all_se_top3[:, 1] / all_se_top3[:, 0], width=0.4, color=och, label='Top3/Full tuned on Top3')

ax.axhline(1., color='grey', linestyle=':', linewidth=1, zorder=-99)

ax.set_xticks(ticks)
ax.set_xticklabels(used_ds, rotation=45, ha='right')

ax.set_ylabel('Relative MSE')

ax.legend()

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(fraction=1.))

se_full_tuned = all_se_full[:, 1] / all_se_full[:, 0]
se_top3_tuned = all_se_top3[:, 1] / all_se_top3[:, 0]

ticks = np.arange(len(used_ds))
ax.bar(ticks - 0.2, se_standard, width=0.2, color=lime, label='Untuned')
ax.bar(ticks, se_full_tuned, width=0.2, color=red, label='Tuned on Full')
ax.bar(ticks + 0.2, se_top3_tuned, width=0.2, color=bluegrey, label='Tuned on Top3')

ax.axhline(1., color='grey', linestyle=':', linewidth=0.65, zorder=-99)

labels = [i[:9] + '...' if len(i) > 9 else i for i in used_ds]

ax.set_xticks(ticks)
ax.set_xticklabels(labels, rotation=90, ha='center')

ax.set_ylabel('Relative SE')
ax.set_xlabel('Dataset')

ax.legend(prop={'size': 10}, loc='center left', bbox_to_anchor=(1, 0.5))
# ax.legend()
# plt.subplots_adjust(bottom=0.15)

fig.tight_layout()
# fig.savefig(f'./Plots/openml/hp_comparison_top3_se.pdf', dpi=500)
# %%
print("Median SE Values:")
print(f"Standard: {np.median(se_standard):.2f}")
print(f"Full Tuned: {np.median(se_full_tuned):.2f}")
print(f"Top3 Tuned: {np.median(se_top3_tuned):.2f}")

#%%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(fraction=1.))

crps_full_tuned = all_crps_full[:, 1] / all_crps_full[:, 0]
crps_top3_tuned = all_crps_top3[:, 1] / all_crps_top3[:, 0]

ticks = np.arange(len(used_ds))
ax.bar(ticks - 0.2, crps_standard, width=0.2, color=lime, label='Untuned')
ax.bar(ticks, crps_full_tuned, width=0.2, color=red, label='Tuned on Full')
ax.bar(ticks + 0.2, crps_top3_tuned, width=0.2, color=bluegrey, label='Tuned on Top3')

ax.axhline(1., color='grey', linestyle=':', linewidth=1, zorder=-99)

ax.set_xticks(ticks)

labels = [i[:9] + '...' if len(i) > 9 else i for i in used_ds]

ax.set_xticklabels(labels, rotation=90, ha='center')

ax.set_ylabel('Relative CRPS')
ax.set_xlabel('Dataset')

ax.legend(prop={'size': 10}, loc='center left', bbox_to_anchor=(1, 0.5))
# ax.legend()

fig.tight_layout()
fig.savefig(f'./Plots/openml/hp_comparison_top3_crps.pdf', dpi=500)
# %%
print("Median CRPS Values:")
print(f"Standard: {np.median(crps_standard):.2f}")
print(f"Full Tuned: {np.median(crps_full_tuned):.2f}")
print(f"Top3 Tuned: {np.median(crps_top3_tuned):.2f}")

# %%
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(fraction=1.5))

ticks = np.arange(len(used_ds))

ax.bar(ticks - 0.2, all_se_top3[:, 1] / all_se_top3[:, 0], width=0.4, color=blue, label='Top3/Full tuned on Top3')
ax.bar(ticks + 0.2,
       all_se_top3[:, 1] / all_se_full[:, 0],
       width=0.4,
       color=och,
       label='Top3 tuned on Top3/Full tuned on Full')

ax.axhline(1., color='grey', linestyle=':', linewidth=1, zorder=-99)

ax.set_xticks(ticks)
ax.set_xticklabels(used_ds, rotation=45, ha='right')

ax.set_ylabel('Relative MSE')

ax.legend()
# %%

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(fraction=1.5))

ticks = np.arange(len(used_ds))

ax.bar(ticks - 0.2, all_crps_top3[:, 1] / all_crps_top3[:, 0], width=0.4, color=blue, label='Top3/Full tuned on Top3')
ax.bar(ticks + 0.2,
       all_crps_top3[:, 1] / all_crps_full[:, 0],
       width=0.4,
       color=och,
       label='Top3 tuned on Top3/Full tuned on Full')

ax.axhline(1., color='grey', linestyle=':', linewidth=1, zorder=-99)

ax.set_xticks(ticks)
ax.set_xticklabels(used_ds, rotation=45, ha='right')

ax.set_ylabel('Relative CRPS')

ax.legend()

# %%
cols = ['max_features', 'min_samples_leaf']

df_all_hp = df_hp_full_crps[cols].join(df_hp_full_se[cols], lsuffix='_full_crps', rsuffix='_full_se')
df_all_hp = df_all_hp.join(df_hp_top3_crps[cols].add_suffix('_top3_crps'))
df_all_hp = df_all_hp.join(df_hp_top3_se[cols].add_suffix('_top3_se'))
df_all_hp
# %%
c = np.sort(df_all_hp.columns)
print(df_all_hp[c].to_latex(float_format="%.2f"))
# %%
