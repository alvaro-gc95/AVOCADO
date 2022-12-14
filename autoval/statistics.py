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
    def __init__(self, data, n_components):
        self.data = self.from_pandas_to_daskarray(data, npartitions=3)
        self.n_components = n_components
        self.anomalies()

    @staticmethod
    def from_pandas_to_daskarray(df: pd.DataFrame, npartitions):
        df = autoval.utils.clean_dataset(df)
        df = dd.from_pandas(df, npartitions=npartitions).to_dask_array()
        return df

    def anomalies(self, standardize=False):
        """
        Calculate anomalies of the field.
        :param standardize: bool, optional. (Default=False). If True standardize of the anomalies.
        """

        mean = self.data.mean()

        if standardize:
            std = self.data.std()
            anomaly = (self.data - mean)/std

        else:
            anomaly = self.data - mean

        return anomaly

    def calculate(self, mode):
        print(self.anomalies(standardize=True))

        # Get Dask array of anomalies
        z = self.anomalies(standardize=True)
        print(z)
        # z = dd.from_pandas(autoval.utils.clean_dataset(self.anomalies(standardize=True)), npartitions=3).to_dask_array()

        if mode == 'T':
            z = z.transpose()
        elif mode == 'S':
            pass
        else:
            raise AttributeError(' Error: ' + mode + ' is not a PCA type')

        # Covariance matrix
        s = da.cov(z.compute_chunk_sizes())

        # Get principal components
        pca = PCA(n_components=self.n_components)
        pca.fit(s)

        # Clear some memory
        del z, s
        gc.collect()
        print(pca.components_.shape)
        print([r * 100 for r in pca.explained_variance_ratio_])




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
