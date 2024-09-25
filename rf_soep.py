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

pd.options.mode.copy_on_write = True

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


# %%
df = pd.read_csv('data/soep/soep_data_Sep9_2024.csv', sep=',')
df["sector"] = df["sector"].fillna("Unknown")
df['sector_id'] = df['sector_id'].fillna(99)
df = df.dropna()
df.shape
# %%
relevant_cols = ['survey_year', 'female', 'age', 'n_persons', 'n_children', 'years_educ', 'employed', 'sector']
y = df['income']  
X = df[relevant_cols]

X = pd.get_dummies(X, columns=['sector'], drop_first=True)
X = pd.get_dummies(X, columns=['employed'], drop_first=True)

train_idx = np.where(X['survey_year'] != 2019)[0]
test_idx = np.where(X['survey_year'] == 2019)[0]

df_train = X.iloc[train_idx]
df_test = X.iloc[test_idx]
y_train = y.iloc[train_idx].values
y_test = y.iloc[test_idx].values

X_train = df_train.values
X_test = df_test.values
# %%
N_TREES = 1000
hyperparams = dict(n_estimators=N_TREES,
                   random_state=SEED,
                   n_jobs=-1,
                   max_features='sqrt',
                   min_samples_split=5,
                   min_samples_leaf=1)

rf = RandomForestWeight(hyperparams=hyperparams, name='rf_soep')
rf.fit(X_train, y_train)

# %%
K = 5
y_hat, w_all = rf.weight_predict(X_test, return_weights=True)

w_k, w_k_sum = get_topk_weight(w_all, k=K, return_sum=True)
y_hat_k = y_train @ w_k.T

mse_full = se(y_test, y_hat)
mse_k = se(y_test, y_hat_k)

crps_full = crps_sample(y_test, y_train, w_all, return_mean=False)
crps_k = crps_sample(y_test, y_train, w_k, return_mean=False)

print(f'MSE full: {mse_full.mean():.2e}, MSE k: {mse_k.mean():.2e}')
print(f'CRPS full: {crps_full.mean():.2f}, CRPS k: {crps_k.mean():.2f}')
#%%
sec_cols = [col for col in df_train if 'sector_' in col]
emp_cols = [col for col in df_train if 'employed_' in col]
non_dummy_cols = [col for col in df_train if 'sector_' not in col]
non_dummy_cols = [col for col in non_dummy_cols if 'employed_' not in col]

tester = df_test.copy(deep=True)
trainer = df_train.copy(deep=True)

tester['Unknown Sector'] = False
trainer['Unknown Sector'] = False

test_idx_unknown = np.where((~tester[sec_cols]).all(1))[0]
train_idx_unknown = np.where((~trainer[sec_cols]).all(1))[0]

test_idx_emp = np.where((~tester[emp_cols]).all(1))[0]
train_idx_emp = np.where((~trainer[emp_cols]).all(1))[0]

tester.loc[tester.index[test_idx_unknown], 'Unknown Sector'] = True
trainer.loc[trainer.index[train_idx_unknown], 'Unknown Sector'] = True

tester.loc[tester.index[test_idx_emp], 'Employed'] = True
trainer.loc[trainer.index[train_idx_emp], 'Employed'] = True

i = 14  #14 male example

w_k_i = w_k[i]
sup_points = np.where(w_k_i > 0)[0]

scen = trainer.iloc[sup_points].copy(deep=True)
scen.loc[:, 'weight'] = w_k_i[sup_points] * 100

scen_sector = []
scen_emp = []
for idx, row in scen.iterrows():
    if all(row[sec_cols] == False):
        # scen.loc[idx, 'Unknown Sector'] = True
        scen_sector.append('Unknown Sector')
    else:
        scen_sector.append(row[sec_cols].idxmax())
    if all(row[emp_cols] == False):
        scen_emp.append('Employed')
    else:
        scen_emp.append(row[emp_cols].idxmax())

scen_sector = list(set(scen_sector))
scen_emp = list(set(scen_emp))

if any(df_test.iloc[i][sec_cols]):
    test_sector = [df_test.iloc[i][sec_cols].idxmax()]
else:
    # df_test['Unknown Sector'] = True
    test_sector = ['Unknown Sector']
if any(df_test.iloc[i][emp_cols]):
    test_emp = [df_test.iloc[i][emp_cols].idxmax()]
else:
    # df_test['Employed'] = True
    test_emp = ['Employed']

scen_cols = non_dummy_cols + scen_emp + scen_sector + ['weight']

display(tester[non_dummy_cols + test_emp + test_sector].iloc[i:i + 1])

print(f'True income: {y_test[i]:.0f}')
print(f'Predicted income: {y_hat[i]:.0f}')
print(f'Topk income: {y_hat_k[i]:.0f}')

print(f'Weight sum: {w_k_sum[i]:.2f}')
print(f'Nonzero weights: {np.count_nonzero(w_all[i]):.0f}')

scen['income'] = y_train[sup_points]
scen_cols = scen_cols + ['income']

scen[scen_cols].sort_values('weight', ascending=False)

#%%
print("Income support points:")
sup_points_w_sorted = sup_points[w_k_i[sup_points].argsort()[::-1]]
# np.round(unstandardize_income(y_train[sup_points_w_sorted]), 0)
np.round(y_train[sup_points_w_sorted], 0)
#%%
print(tester[non_dummy_cols + test_emp + test_sector].iloc[i:i + 1].to_latex(float_format='%.1f', index=True))
print(scen[scen_cols].sort_values('weight', ascending=False).to_latex(float_format='%.1f', index=True))


#%%
# Performance run for different k

k_arr = [3, 5, 10, 20, 50]

results = []

results.append({'mse': mse_full.mean(), 'crps': crps_full.mean()})

for k in k_arr:

    w_k = get_topk_weight(w_all, k=k, return_sum=False)
    y_hat_k = y_train @ w_k.T

    mse_k = se(y_test, y_hat_k)
    crps_k = crps_sample(y_test, y_train, w_k, return_mean=False)

    results.append({
        'mse': mse_k.mean(),
        'rel_mse': mse_k.mean() / mse_full.mean(),
        'crps': crps_k.mean(),
        'rel_crps': crps_k.mean() / crps_full.mean()
    })

results_df = pd.DataFrame(results, index=['full'] + k_arr)
print(results_df.to_latex(float_format='%.2f', index=True))
print(results_df.to_latex(float_format='%.2e', index=True))


# %%
