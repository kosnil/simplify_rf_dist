# Code to the Paper "Simplifying Random Forests' Probabilistic Forecasts"

This repository contains the implementation of experiments from the paper titled "Simplifying Random Forests' Probabilistic Forecasts". 

**Abstract** Since their introduction by Breiman, Random Forests (RFs) have proven to be useful for both classification and regression tasks. 
The RF prediction of a previously unseen observation can be represented as a weighted sum of all training sample observations. 
This nearest-neighbor-type representation is useful, among other things, for constructing forecast distributions (Meinshausen, 2006). 
In this paper, we consider simplifying RF-based forecast distributions by sparsifying them. That is, we focus on a small subset of nearest neighbors while setting the remaining weights to zero. 
This sparsification step greatly improves the interpretability of RF predictions. It can be applied to any forecasting task without re-training existing RF models. 
In empirical experiments, we document that the simplified predictions can be similar to or exceed the original ones in terms of forecasting performance. 
We explore the statistical sources of this finding via a stylized analytical model of RFs. The model suggests that simplification is particularly promising if the unknown true forecast distribution contains many small weights that are estimated imprecisely. 

You can find the paper on arXiv [here](https://arxiv.org/abs/2408.12332).
