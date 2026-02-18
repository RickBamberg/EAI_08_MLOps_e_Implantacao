# src/features/v1/rename_features.py

from sklearn.base import BaseEstimator, TransformerMixin

class RenameColumns(BaseEstimator, TransformerMixin):
    def __init__(self, mapping: dict):
        self.mapping = mapping

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        return X.rename(columns=self.mapping)
