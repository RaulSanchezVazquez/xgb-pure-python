#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 12:36:17 2018

@author: raulsanchez
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 11:38:22 2018

@author: raulsanchez
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import plot_tree
import matplotlib.pyplot as plt

import datasets

(X, y) = datasets.get_data('boston')

reg_lambda = 1
base_score = 0
learning_rate = 1
max_depth = 1
n_estimators = 1

model = xgb.XGBRegressor(
    tree_method='exact',
    objective='reg:linear',
    base_score=base_score,
    learning_rate=learning_rate,
    reg_lambda=reg_lambda,
    max_depth=max_depth,
    n_estimators=n_estimators,
    random_state=0)

model.fit(X, y)
fig, ax = plt.subplots(1, 1, figsize=(6, 6))
ax = plot_tree(model, num_trees=0, ax=ax)

# Split finding
g_i = y - base_score
h_i = pd.Series([1] * y.shape[0])

gain = 0
G = g_i.sum()
H = h_i.sum()

score = 0
best_splits = []

fig, ax = plt.subplots(1, 1)
for k in X.columns:
    G_L, H_L = 0, 0
    local_scores = []
    for j_idx, j in X.sort_values(k)[[k]].iterrows():
        G_L += g_i[j_idx]; H_L += h_i[j_idx]
        G_R = G - G_L; H_R = H - H_L
        
        local_score = (
            G_L**2 / (H_L + reg_lambda)
        ) + (
            G_R**2 / (H_R + reg_lambda)
        ) - (
            G**2 / (H + reg_lambda)
        )
        
        local_scores.append({
            'index': j[k],
            k: local_score})
    
    local_scores = pd.DataFrame(
        local_scores
    ).set_index('index')[k]
    
    feature_value = local_scores[
        local_scores.idxmax()
    ].index[0]
    
    score_max = local_scores.max()
    best_splits.append([feature_value, score_max])
        
    local_scores.plot(
        grid=True,
        legend=True,
        ax=ax)
    
# Optimal weight
feature = 'LSTAT'
th = 7.5
L = y[X[feature] < th]
R = y[X[feature] >= th]

print("Left Leaf:" , np.round(-1 * (base_score - L).sum() / (L.shape[0] + reg_lambda), 4))
print("Right Leaf:" , np.round(-1 * (base_score - R).sum() / (R.shape[0] + reg_lambda), 4))