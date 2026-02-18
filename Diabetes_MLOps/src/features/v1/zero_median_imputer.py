# src/features/v1/zero_median_imputer.py

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class ZeroMedianImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns
        self.medians_ = {}

    def fit(self, X, y=None):
        X = X.copy()
        for col in self.columns:
            median = X.loc[X[col] > 0, col].median()
            self.medians_[col] = median
        return self

    def transform(self, X):
        X = X.copy()
        for col, median in self.medians_.items():
            X.loc[X[col] == 0, col] = median
        return X
