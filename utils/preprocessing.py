import pandas as pd
import numpy as np

class RobustLabelEncoder:
    """
    A robust label encoder that maps categorical string values to integers
    and handles unseen categories gracefully during inference by mapping them
    to a default 'unknown' category.
    """
    def __init__(self, default_val='unknown'):
        self.default_val = default_val
        self.mapping = {}
        self.inverse_mapping = {}

    def fit(self, series):
        # Extract unique values and convert them to string
        unique_vals = list(series.astype(str).unique())
        
        # Add default_val if not already present
        if self.default_val not in unique_vals:
            unique_vals.append(self.default_val)
            
        # Create mapping dictionary
        self.mapping = {val: idx for idx, val in enumerate(unique_vals)}
        self.inverse_mapping = {idx: val for idx, val in enumerate(unique_vals)}
        return self

    def transform(self, series):
        # Map values to their index, fallback to default_val index if unseen
        default_idx = self.mapping.get(self.default_val, 0)
        return series.astype(str).map(lambda x: self.mapping.get(x, default_idx))

    def fit_transform(self, series):
        self.fit(series)
        return self.transform(series)

    def inverse_transform(self, series):
        default_val = self.default_val
        return series.map(lambda x: self.inverse_mapping.get(x, default_val))
