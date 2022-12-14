"""

"""

import itertools

import numpy as np
import xarray as xr
import gc
import dask.array as da
import dask.dataframe as dd
from dask_ml.decomposition import PCA
import pandas as pd
from sklearn.linear_model import LinearRegression

import autoval.utils


class DaskPCA:
    def __init__(self, data, n_components, standardize=False):
        self.data = data
        self.n_components = n_components
        self.anomaly = self.anomalies(standardize=standardize)

    @staticmethod
    def from_pandas_to_daskarray(df: pd.DataFrame, npartitions):
        df = autoval.utils.clean_dataset(df)
        daskarr = dd.from_pandas(df, npartitions=npartitions).to_dask_array()
        daskarr = daskarr.compute_chunk_sizes()

        return daskarr

    def anomalies(self, standardize=False):
        """
        Calculate anomalies of the field.
        :param standardize: bool, optional. (Default=False). If True standardize of the anomalies.
        """

        mean = self.data.mean()

        if standardize:
            std = self.data.std()
            anomaly = (self.data - mean) / std

        else:
            anomaly = self.data - mean

        return anomaly

    def calculate(self, mode):

        # Get Dask array of anomalies
        z = self.from_pandas_to_daskarray(self.anomaly, npartitions=3)

        # PCA mode
        if mode == 'T':
            z = z.transpose()
        elif mode == 'S':
            pass
        else:
            raise AttributeError(' Error: ' + mode + ' is not a PCA type')

        # Covariance matrix
        s = da.cov(z)

        # Get principal components
        pca = PCA(n_components=self.n_components)
        pca.fit(s)

        pc = z.dot(pca.components_)
        # Clear some memory
        del z, s
        gc.collect()

        print(pc)
        print(pca.components_)
        print([r * 100 for r in pca.explained_variance_ratio_])

        return pca


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
