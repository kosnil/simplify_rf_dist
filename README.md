# Code to the Paper "Simplifying Random Forests' Probabilistic Forecasts"

This repository contains the implementation of experiments from the paper titled "Simplifying Random Forests' Probabilistic Forecasts".  

You can find the paper on arXiv [here](https://arxiv.org/abs/2408.12332).



## Abstract
Since their introduction by Breiman, Random Forests (RFs) have proven to be useful for both classification and regression tasks.
The RF prediction of a previously unseen observation can be represented as a weighted sum of all training sample observations.
This nearest-neighbor-type representation is useful, among other things, for constructing forecast distributions [(Meinshausen, 2006)](https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf).
In this paper, we consider simplifying RF-based forecast distributions by sparsifying them. That is, we focus on a small subset of nearest neighbors while setting the remaining weights to zero.
This sparsification step greatly improves the interpretability of RF predictions. It can be applied to any forecasting task without re-training existing RF models.
In empirical experiments, we document that the simplified predictions can be similar to or exceed the original ones in terms of forecasting performance.
We explore the statistical sources of this finding via a stylized analytical model of RFs. The model suggests that simplification is particularly promising if the unknown true forecast distribution contains many small weights that are estimated imprecisely.



## Dependencies
This repository contains the code used to generate the results in the paper. The code is written in Python and uses the libraries (as specified in `requirements.txt`):
- `joblib==1.4.0`
- `matplotlib==3.8.4`
- `numba==0.59.1`
- `numpy==2.2.6`
- `pandas==2.2.3`
- `scikit_learn==1.4.2`
- `scipy==1.15.3`
- `seaborn==0.13.2`
- `tqdm==4.66.2`


## Usage

This repository contains the code to replicate the results in the paper (including training, tuning and evaluation) and to apply Topk to your own data. The main code can be found in the `RF.py` file. The code is organized in a modular way, so you can easily adapt it to your own needs. The main class is `RandomForestWeight`, which is a wrapper around the `RandomForestRegressor` class from `sklearn`. The class contains methods to train the model, predict the test set, and calculate the weights (independent of the choice of $k$, as a byproduct we also implement Meinshausen's Quantile Regression Forests).  
A first starting point is the `minimal_working_example.ipynb` notebook, which contains a minimal working example of how to use the code. The notebook contains a step-by-step guide to replicate the results in the paper.
The basic workflow is as follows:

```python
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression

# Set seed for reproducibility
SEED = 7531
np.random.seed(SEED)

# Load dataset (in this case, we create a synthetic dataset)
X, y = make_regression(n_samples=5000, n_features=20, noise=1, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)

# Create a Topk RF class and train. The RandomForestWeight class is based on the RandomForestRegressor class from sklearn.
from RF import RandomForestWeight

# define hyperparameters for Topk RF
hyperparams = dict(
    n_estimators=1000,
    random_state=SEED,
    n_jobs=-1,
    max_features='sqrt',
    min_samples_split=5,
)

rf = RandomForestWeight(hyperparams=hyperparams)
rf.fit(X_train, y_train)

# Predict the test set
k = 5
y_hat_k, w_k = rf.weight_predict(X_test, top_k=k, return_weights=True)
```

To reproduce the results in the paper, you can run the scripts in the files `rf_restrict_k_openml.py`, `rf_hp_tuning.py`, `tuned_score_comparison` and `rf_soep.py`. These scripts contain the code to train, tune, and evaluate models on the OpenML datasets as well as the SOEP dataset. Results are stored in the `results/` directory.

To be as efficient as possible, we calculate the weights in parallel using numba. 
As these calculations take place in-memory, this can lead to memory issues for larger datasets. To avoid this, we recommend using the `sparse` versions of the functions. We refer to `minimal_working_example.ipynb` for details.


## Directory Structure
```
simplify_rf_dist/  
├── [README.md](README.md)    
├── [requirements.txt](requirements.txt)  
├── data/        
│   ├── soep_prep/          
│   │   ├── [prepare_soep_data.R](data/soep_prep/prepare_soep_data.R)  
│   └── ...                
├── utils/                  
│   ├── [plotting_helpers.py](utils/plotting_helpers.py)  
│   ├── [score_utils.py](utils/score_utils.py)  
│   ├── [sparse_utils.py](utils/sparse_utils.py)  
│   └── ...                 
├── results/                  
├── Plots/                  
├── weight_storage/         
├── [minimal_working_example.ipynb](minimal_working_example.ipynb)  
├── [RF.py](RF.py)                   
├── [rf_hp_tuning.py](rf_hp_tuning.py)         
├── [rf_restrict_k_openml.py](rf_restrict_k_openml.py)  
├── [rf_soep.py](rf_soep.py)              
├── [tuned_score_comparison.py](tuned_score_comparison.py)  
├── Theoretical Example/    
│   ├── [Toyexample.ipynb](Theoretical%20Example/Toyexample.ipynb)   
│   └── ...                 
└── [LICENSE](LICENSE)                
```



<!--
# Load California housing dataset from OpenML
ca_housing_id = 44138
ca_housing_ds = openml.datasets.get_dataset(ca_housing_id)

X, y, _, _ = ds.get_data(target=ds.default_target_attribute, dataset_format='dataframe')
df_train, df_test, y_train, y_test = train_test_split(X, y.values, test_size=0.3, random_state=SEED)

# Convert the dataframes to numpy arrays
X_train = df_train.values
X_test = df_test.values
y_train = y_train.astype(np.float32)
y_test = y_test.astype(np.float32)
-->