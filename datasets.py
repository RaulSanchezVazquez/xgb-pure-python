#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 11:11:32 2018

@author: raulsanchez
"""

import pandas as pd
from sklearn import datasets

def get_data(dataname='boston'):
    if (dataname == 'boston'):
        data = datasets.load_boston()
    
    if (dataname == 'iris'):
        data = datasets.load_iris()
        
    X = pd.DataFrame(
        data['data'],
        columns=data['feature_names'])
    X = X[list(reversed(X.columns))]
    
    y = pd.Series(data['target'])
    
    X = X.round(0)
    
    return (X, y)
