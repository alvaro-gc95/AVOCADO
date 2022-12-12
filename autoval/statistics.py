"""

"""

import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

import autoval.utils


class xPCA:
    def __init__(self, data):
        self.data = data

    def anomalies(self, kind, *months):

        anomalies = pd.DataFrame(index=self.data.index, columns=[c + '_anomaly' for c in self.data.columns])

        if kind == 'monthly':
            for month, monthly_df in self.data.groupby(self.data.index.month):
                monthly_mean = monthly_df.mean()
                anomaly = monthly_df - monthly_mean
                anomalies.loc[anomaly.index] = anomaly

        elif kind == 'seasonal':
            mean = self.data.resample('M').mean()

        elif kind == 'total':
            mean = self.data.mean()

        elif kind == 'custom':
            mean = self.data.resample('M').mean()


def linear_regression(x: pd.DataFrame, y: pd.DataFrame):
    # Clean DataFrames of possible conflictive values
    x = autoval.utils.clean_dataset(x)
    y = autoval.utils.clean_dataset(y)

    # Get only the common data
    common_idx = list(set(x.index).intersection(y.index))
    x = x.loc[common_idx]
    y = y.loc[common_idx]

    # Create object for the class
    linear_regressor = LinearRegression()

    # Perform linear regression
    linear_regressor.fit(x, y)
    residuals = x - linear_regressor.predict(x)

    return linear_regressor, residuals
