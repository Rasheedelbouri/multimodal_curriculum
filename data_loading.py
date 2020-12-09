# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 18:21:38 2020

@author: rashe
"""
from sklearn.datasets import load_iris, load_digits, load_boston, load_breast_cancer, load_diabetes
import pandas as pd


def loadData(dataset):
    
    if not isinstance(dataset, str):
        raise("input a valid argument")
    
    if dataset.lower() == 'iris':
        func = load_iris
    elif dataset.lower() == 'digits':
        func = load_digits
    elif dataset.lower() == 'boston':
        func = load_boston
    elif dataset.lower() == "breast_cancer":
        func = load_breast_cancer
    elif dataset.lower == "diabetes":
        func = load_diabetes
    else:
        raise("can't find this dataset")
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    X,y = func(return_X_y=True)
    unique_classes = pd.DataFrame(y)[0].unique()
    if dataset.lower() in ('iris', 'digits', 'breast_cancer'):
        ohe = OneHotEncoder(sparse=False)
        y = ohe.fit_transform(pd.DataFrame(y))
    train_x, val_x, train_y, val_y = train_test_split(X,y,train_size=0.6)
    val_x, test_x, val_y, test_y = train_test_split(val_x, val_y,train_size=0.5)
    
    return train_x, train_y, val_x, val_y, test_x, test_y, unique_classes
