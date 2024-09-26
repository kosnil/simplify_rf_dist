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
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
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
import gc

TEST_SIZE = 0.3
NO_PROCESSES = 1

TOP_K_MAX = 200
N_TREES = 1000

HP_TUNING = False
BAGGED_TREES = True
GRID_SEARCH = True

STANDARDIZE_Y = False

SHALLOW = False

for i, ds in enumerate(paper_ds):

    if HP_TUNING is True:
        if BAGGED_TREES is False:
            file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}_hptuned_noBT.pkl"
        elif GRID_SEARCH is True:
            file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}_hptuned_grid.pkl"
        else:
            file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}_hptuned.pkl"
    elif SHALLOW is True:
        file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}_shallow.pkl"
    else:
        if STANDARDIZE_Y is True:
            file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}_standardized.pkl"
        else:
            file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}.pkl"

    if os.path.exists(file_name):
        print("Skipping", ds.name, "as it is already processed")
        continue

    print(ds.name, 'target:', ds.default_target_attribute)

    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")

    if STANDARDIZE_Y:
        y_mean = y.mean()
        y_std = y.std()
        y = (y - y_mean) / y_std

    if ds.name == 'sulfur':
        X = X.drop(columns=['y2'])

    if ds.name == 'delays_zurich_transport':
        if SHALLOW is True:
            if os.path.exists(cwd + f'/data/delays_zurich_transport_randixs004.pkl'):
                with open(cwd + f'/data/delays_zurich_transport_randixs004.pkl', 'rb') as file:
                    rand_idxs = pickle.load(file)
            else:
                rand_idxs = np.random.choice(len(X), size=int(len(X) * 0.04), replace=False)
                with open(cwd + f'/data/delays_zurich_transport_randixs004.pkl', 'wb') as file:
                    pickle.dump(rand_idxs, file)
        else:
            if os.path.exists(cwd + f'/data/delays_zurich_transport_randixs02.pkl'):
                with open(cwd + f'/data/delays_zurich_transport_randixs02.pkl', 'rb') as file:
                    rand_idxs = pickle.load(file)
            else:
                rand_idxs = np.random.choice(len(X), size=int(len(X) * 0.2), replace=False)
                with open(cwd + f'/data/delays_zurich_transport_randixs02.pkl', 'wb') as file:
                    pickle.dump(rand_idxs, file)
        X = X.iloc[rand_idxs]
        y = y.iloc[rand_idxs]

    if SHALLOW is True and ds.name == 'nyc-taxi-green-dec-2016':
        rand_idxs = np.random.choice(len(X), size=int(len(X) * 0.3), replace=False)
        X = X.iloc[rand_idxs]
        y = y.iloc[rand_idxs]

    df_train, df_test, y_train, y_test = train_test_split(X, y.values, test_size=TEST_SIZE, random_state=SEED)

    X_train = df_train.values
    X_test = df_test.values

    print(f'Dataset split. X_train shape: {X_train.shape}, X_test shape: {X_test.shape}')

    sparse_loop = True if len(y) > 150_000 else False

    del X, y
    del df_train, df_test

    gc.collect()

    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    if HP_TUNING is True:

        rf = RandomForestRegressor(criterion='squared_error', n_jobs=12 if len(y_train) > 100_000 else -1)

        hp_params = {
            'random_state': [SEED],
            'max_depth': [None],
            'min_samples_leaf': [1, 2, 4, 6, 8, 10, 15, 20, 30, 40, 50],
            'min_samples_split': [5],
            'n_estimators': [1000],
            'max_features': [0.333, 'sqrt', 0.5, 1.0] if BAGGED_TREES is True else [0.333, 'sqrt', 0.5],
        }

        print("Starting hyperparameter tuning...")

        hp_filename = cwd + f"/results/openml/hyperparams_{ds.name.replace(' ', '_')}_bagged{BAGGED_TREES}_grid{GRID_SEARCH}.pkl"

        if os.path.exists(hp_filename):
            with open(hp_filename, 'rb') as file:
                hyperparams = pickle.load(file)

            print('Loaded hyperparams:', hyperparams)
        else:

            cv_jobs = 12 if len(y_train) < 100_000 else 2
            no_cv = 5 if len(y_train) < 100_000 else 3

            if GRID_SEARCH is True:
                clf = GridSearchCV(rf,
                                   hp_params,
                                   n_jobs=cv_jobs,
                                   verbose=True,
                                   cv=no_cv,
                                   scoring='neg_mean_squared_error')
            else:
                clf = RandomizedSearchCV(rf,
                                         hp_params,
                                         random_state=SEED,
                                         n_jobs=cv_jobs,
                                         n_iter=100,
                                         verbose=True,
                                         cv=no_cv,
                                         scoring='neg_mean_squared_error')
            search = clf.fit(X_train, y_train)
            hyperparams = search.best_params_

            with open(
                    cwd +
                    f"/results/openml/hyperparams_{ds.name.replace(' ', '_')}_bagged{BAGGED_TREES}_grid{GRID_SEARCH}.pkl",
                    "wb") as file:
                pickle.dump(hyperparams, file)

            del rf
            del clf
            del search

            print('Done with hyperparameter tuning. Best params: ', hyperparams)

    else:
        hyperparams = dict(n_estimators=N_TREES,
                           random_state=SEED,
                           n_jobs=-1,
                           max_features='sqrt',
                           min_samples_split=5,
                           min_samples_leaf=1)

        if SHALLOW is True:
            hyperparams['min_samples_leaf'] = 50

    print("Train RF with hyperparams: ", hyperparams)
    rf = RandomForestWeight(hyperparams=hyperparams, name='rf_' + ds.name)
    rf.fit(X_train, y_train)

    print("RF trained...")

    if sparse_loop is True:
        gc.collect()

        if HP_TUNING is True:
            if GRID_SEARCH is True:
                weight_path = cwd + '/weight_storage/rf_' + ds.name + '_weights_hptuned_grid.npz'
            else:
                weight_path = cwd + '/weight_storage/rf_' + ds.name + '_weights_hptuned.npz'

        elif SHALLOW is True:
            weight_path = cwd + '/weight_storage/rf_' + ds.name + '_weights_shallow.npz'
        else:
            if STANDARDIZE_Y is True:
                weight_path = cwd + '/weight_storage/rf_' + ds.name + '_weights_standardized.npz'
            else:
                weight_path = cwd + '/weight_storage/rf_' + ds.name + '_weights.npz'

        if os.path.exists(weight_path):
            w_all_sparse = load_npz(weight_path).tocsr()
            print("Weights loaded...")
        else:
            w_all_sparse = rf.get_rf_weights_sparse(X_test, sort=True)
            save_npz(weight_path, w_all_sparse)
            print("Weights calculated and stored...")

        gc.collect()

        results_ds = topk_looper_sparse(X_test=X_test,
                                        y_train=y_train,
                                        y_test=y_test,
                                        rf=rf,
                                        w_hat=w_all_sparse,
                                        k_max=TOP_K_MAX,
                                        k_stepsize=1,
                                        verbose=True)

    else:
        if len(y_test) < 15000:
            NO_PROCESSES = 2
        elif len(y_test) < 10000:
            NO_PROCESSES = 4
        elif len(y_test) < 5000:
            NO_PROCESSES = 8

        gc.collect()

        results_ds = topk_looper(X_test=X_test,
                                 y_train=y_train,
                                 y_test=y_test,
                                 rf=rf,
                                 k_max=TOP_K_MAX,
                                 num_processes=NO_PROCESSES,
                                 verbose=True)

    if HP_TUNING is True:
        results_ds['hyperparams'] = hyperparams

    with open(file_name, "wb") as file:
        pickle.dump(results_ds, file)

    del results_ds

# %%
results = []
used_ds = []

considered_ks = [3, 5, 10, 20, 50]

HP_TUNED_RESULTS = True
SHALLOW_RESULTS = False
GRID_SEARCH_RESULTS = True

STANDARDIZE_Y_RESULTS = True

for ds in paper_ds:
    if HP_TUNED_RESULTS is True:
        if BAGGED_TREES is False:
            file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}_hptuned_noBT.pkl"
        elif GRID_SEARCH_RESULTS is True:
            file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}_hptuned_grid.pkl"
        else:
            file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}_hptuned.pkl"

    elif SHALLOW_RESULTS is True:
        file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}_shallow.pkl"
    else:
        if STANDARDIZE_Y_RESULTS is True:
            file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}_standardized.pkl"
        else:
            file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}.pkl"
    if os.path.exists(file_name):
        used_ds.append(ds.name)
        with open(file_name, "rb") as file:
            result = pickle.load(file)
            results.append(result)
# %%

fig, ax = plt.subplots(nrows=6, ncols=3, sharex=False, figsize=(17, 30))

METRIC = 'crps'
METRIC_LABEL = METRIC.upper() if METRIC == 'crps' else 'M' + METRIC.upper()

counter = 0
for row in ax:
    for col in row:

        if METRIC == 'topk_sums':
            if results[counter][METRIC].shape[0] >= 200:
                to_plot = results[counter][METRIC].mean(1)
            else:
                to_plot = np.interp(np.arange(0, 201, 1)[1:], np.arange(1, 200, 2), results[counter][METRIC].mean(1))
            col.plot(to_plot, color=blue)
            col.axhline(1.0, ls='--', color='grey')
            col.set_ylim(-0.05, 1.05)
            col.set_title(f"Sum(Topk) Dataset No.{counter}: {used_ds[counter]}")
        else:
            if results[counter][METRIC].shape[0] == 201:
                to_plot = results[counter][METRIC][:-1].mean(1)
            else:
                to_plot = np.interp(
                    np.arange(0, 201, 1)[1:], np.arange(1, 200, 2), results[counter][METRIC].mean(1)[:-1])
            col.plot(to_plot, color=blue)
            col.axhline(results[counter][METRIC][-1].mean(), ls='--', color='grey')
            col.set_title(f"{METRIC_LABEL.upper()} Dataset No.{counter}: {used_ds[counter]}")

        col.plot()
        col.set_xticks(np.arange(0, TOP_K_MAX + 1, 20))

        counter += 1

        # if counter == len(results):
        #     fig.delaxes(ax[-1][-2])
        #     fig.delaxes(ax[-1][-1])
        #     break
# %%

considered_ks = [3, 5, 10, 20, 50]
loss = 'crps'
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

print(df_results[[col for col in df_results.columns if "Skill" not in col]].to_latex(float_format="%.4f"))
#%%
skill_cols = [col for col in df_results.columns if "Rel" in col]
df_results[skill_cols] = df_results[skill_cols].astype(float)

print(df_results[skill_cols].median())

skill_cols = [f'{loss_label} Full'] + skill_cols  #+ ['Best k']

format_map = {
    f'{loss_label} Full': "{:.4f}",
    f'{loss_label} Top3 Rel': "{:.2f}",
    f'{loss_label} Top5 Rel': "{:.2f}",
    f'{loss_label} Top10 Rel': "{:.2f}",
    f'{loss_label} Top20 Rel': "{:.2f}",
    f'{loss_label} Top50 Rel': "{:.2f}",
    'Best k': "{}",
}
df_styled = df_results[skill_cols].style.format(format_map)

print(df_styled.to_latex())
#%%
df_styled

#%%
unconditional_crps = []

for i, ds in enumerate(paper_ds):

    print(ds.name, 'target:', ds.default_target_attribute)

    X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format="dataframe")

    if ds.name == 'sulfur':
        X = X.drop(columns=['y2'])

    if ds.name == 'delays_zurich_transport':
        rand_idxs = np.random.choice(len(X), size=int(len(X) * 0.2), replace=False)
        X = X.iloc[rand_idxs]
        y = y.iloc[rand_idxs]

    df_train, df_test, y_train, y_test = train_test_split(X, y.values, test_size=TEST_SIZE, random_state=SEED)

    X_train = df_train.values
    X_test = df_test.values

    del X, y
    del df_train, df_test

    gc.collect()

    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    unc_crps = crps_sample_unconditional(y_test, dat=y_train)

    unconditional_crps.append(unc_crps)

unconditional_crps = np.array([unc_crps.mean() for unc_crps in unconditional_crps])
#%%
unconditional_crps_ratio = unconditional_crps / df_results['CRPS Full'].values
df_unc_crps = pd.DataFrame(index=used_ds, data=unconditional_crps_ratio, columns=['Unconditional CRPS'])
#%%
ratio = []

for i, ds in enumerate(used_ds):
    ae_i = results[i]['ae'][-1].mean()
    crps_i = results[i]['crps'][-1].mean()
    ratio.append(ae_i / crps_i)

df_ratio_mae_crps = pd.DataFrame(index=used_ds, data=ratio, columns=['Ratio MAE/CRPS'])
df_ratio_mae_crps
#%%
df_ratio_top3 = pd.merge(df_ratio_mae_crps, df_results['CRPS Top3 Rel'], left_index=True, right_index=True)
df_ratio_top3 = pd.merge(df_ratio_top3, df_unc_crps, left_index=True, right_index=True)
df_ratio_top3.index = [i[:9] + '...' if len(i) > 9 else i for i in df_ratio_top3.index]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=set_size(fraction=1.))
df_ratio_top3[['Ratio MAE/CRPS', 'Unconditional CRPS', 'CRPS Top3 Rel']].plot(kind='bar',
                                                                              color=[blue, och, lime],
                                                                              rot=90,
                                                                              ax=ax)
ax.axhline(1., ls=':', color='grey', zorder=-1, linewidth=0.65)

ax.set_ylabel("Relative CRPS")
ax.set_xlabel("Dataset")

ax.set_yscale('log')
ax.set_ylim(0, 19.99)
ax.set_yticks([1, 1.5, 2, 5, 10, 15])

from matplotlib.ticker import ScalarFormatter

ax.yaxis.set_major_formatter(ScalarFormatter())

blue_patch = mpatches.Patch(color=blue, label='Point Forecast')
och_patch = mpatches.Patch(color=och, label='Unconditional')
lime_patch = mpatches.Patch(color=lime, label='Top3')

ax.legend(handles=[blue_patch, och_patch, lime_patch], prop={'size': 10}, loc='center left', bbox_to_anchor=(1, 0.5))

fig.tight_layout()
fig.savefig(f'./Plots/openml/fullmaecrpsratio_top3crps_dotted.pdf', dpi=500)
#%%
if HP_TUNED_RESULTS is True:

    format_map = {
        'max_depth': "{}",
        'min_samples_leaf': "{:d}",
        'min_samples_split': "{:d}",
        'n_estimators': "{:d}",
        'max_features': "{}"
    }

    df_hps = {}
    for i, ds in enumerate(used_ds):
        df_hps[ds] = results[i]['hyperparams']

    df_hps = pd.DataFrame(df_hps).T

    df_hps = df_hps.astype({'min_samples_leaf': 'int32', 'min_samples_split': 'int32', 'n_estimators': 'int32'})

    df_hps_styled = df_hps[['max_depth', 'min_samples_leaf', 'min_samples_split', 'n_estimators',
                            'max_features']].style.format(format_map)

    print(df_hps_styled.to_latex())

#%%
df_sumtopk = {}

loss = 'topk_sums'

for i, ds in enumerate(used_ds):
    df_sumtopk[ds] = {}
    for k in considered_ks:
        df_sumtopk[ds][f'Sum Top{k}'] = results[i][loss][k - 1].mean()
    df_sumtopk[ds]['n_train'] = int(df_overview.loc[ds]['Length'] *
                                    (1 - TEST_SIZE)) if ds != 'delays_zurich_transport' else int(
                                        df_overview.loc[ds]['Length'] * 0.2 * (1 - TEST_SIZE))

    best_k_crps = np.argmin(results[i]['crps'].mean(1))
    best_k_mse = np.argmin(results[i]['se'].mean(1))

df_sumtopk = pd.DataFrame(df_sumtopk).T

format_map = {
    'Sum Top3': "{:.3f}",
    'Sum Top5': "{:.3f}",
    'Sum Top10': "{:.3f}",
    'Sum Top20': "{:.3f}",
    'Sum Top50': "{:.3f}",
    'n_train': "{:.0f}",
}

print(df_sumtopk.style.format(format_map).to_latex())
#%%
results = []
results_hptuned = []
used_ds = []

considered_ks = [3, 5, 10, 20, 50]

HP_TUNED_RESULTS = True
SHALLOW_RESULTS = False

for ds in paper_ds:
    if BAGGED_TREES is False:
        file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}_hptuned_noBT.pkl"
    else:
        file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}_hptuned.pkl"

    if os.path.exists(file_name):
        used_ds.append(ds.name)
        with open(file_name, "rb") as file:
            result = pickle.load(file)
            results_hptuned.append(result)

    file_name = cwd + f"/results/openml/results_{ds.name.replace(' ', '_')}.pkl"
    if os.path.exists(file_name):
        with open(file_name, "rb") as file:
            result = pickle.load(file)
            results.append(result)
#%%
df_sumtopk = {}
df_sumtopk_hp = {}

loss = 'topk_sums'

for i, ds in enumerate(used_ds):
    df_sumtopk[ds] = {}
    df_sumtopk_hp[ds] = {}
    for k in considered_ks:
        df_sumtopk[ds][f'Sum Top{k}'] = results[i][loss][k - 1].mean()
        df_sumtopk_hp[ds][f'Sum Top{k}'] = results_hptuned[i][loss][k - 1].mean()
    df_sumtopk[ds]['n_train'] = int(df_overview.loc[ds]['Length'] *
                                    (1 - TEST_SIZE)) if ds != 'delays_zurich_transport' else int(
                                        df_overview.loc[ds]['Length'] * 0.2 * (1 - TEST_SIZE))

    best_k_crps = np.argmin(results[i]['crps'].mean(1))
    best_k_mse = np.argmin(results[i]['se'].mean(1))

df_sumtopk = pd.DataFrame(df_sumtopk).T
df_sumtopk_hp = pd.DataFrame(df_sumtopk_hp).T

format_map = {
    'Sum Top3': "{:.3f}",
    'Sum Top5': "{:.3f}",
    'Sum Top10': "{:.3f}",
    'Sum Top20': "{:.3f}",
    'Sum Top50': "{:.3f}",
    'n_train': "{:.0f}",
}
