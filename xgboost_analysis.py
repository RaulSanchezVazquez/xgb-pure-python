#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 27 11:38:22 2018

@author: raulsanchez
"""
import numpy as np
import xgboost as xgb
from xgboost import plot_tree
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split

data = datasets.load_boston()

X = pd.DataFrame(
    data['data'],
    columns=data['feature_names'])

y = pd.Series(data['target'])

X_train, X_test, y_train, y_test =  train_test_split(
    X, y,
    train_size=.8,
    random_state=0)

reg_lambda = 0
base_score=2
model = xgb.XGBRegressor(
    base_score=base_score,
    tree_method='exact',
    objective='reg:linear',
    learning_rate=1,
    reg_lambda=reg_lambda,
    max_depth=1,
    random_state=0,
    n_estimators=1)

model.fit(X_train, y_train)
plot_tree(model, num_trees=0)

feature = 'LSTAT'
th = 8.13
L = y_train[X_train[feature] < th]
R = y_train[X_train[feature] >= th]

print("Left Leaf:" , np.round(-1 * (base_score - L).sum() / (L.shape[0] + reg_lambda), 4))
print("Right Leaf:" , np.round(-1 * (base_score - R).sum() / (R.shape[0] + reg_lambda), 4))