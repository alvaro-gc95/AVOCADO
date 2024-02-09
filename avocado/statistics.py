"""
Statistical methods

Contact: alvaro@intermet.es
"""

import gc
import dask.array as da
import dask.dataframe as dd
from dask_ml.decomposition import PCA
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

import avocado.utils


class DaskPCA:
    def __init__(self, data, n_components, mode, standardize=False):
        self.data = data
        self.n_components = n_components
        self.mean = self.data.mean()
        self.mode = mode
        self.standardize = standardize
        self.anomaly = self.anomalies(standardize=standardize)
        self.eof, self.pc, self.explained_variance = self.calculate(mode=mode)

    @staticmethod
    def from_pandas_to_daskarray(df: pd.DataFrame, npartitions):
        df = avocado.utils.clean_dataset(df)
        daskarr = dd.from_pandas(df, npartitions=npartitions).to_dask_array()
        daskarr = daskarr.compute_chunk_sizes()

        return daskarr

    def anomalies(self, standardize=False):
        """
        Calculate anomalies of the field.
        :param standardize: bool, optional. (Default=False). If True standardize of the anomalies.
        """

        if standardize:
            std = self.data.std()
            anomaly = (self.data - self.mean) / std

        else:
            anomaly = self.data - self.mean

        return anomaly

    def calculate(self, mode):
        """
        Calculate the principal components, empirical orthogonal functions and explained variance ratio.
        :param mode: str. options = 'S' or 'T'. Mode of Analysis.
        :return eofs, pc, explained_variance: (DataFrame, DataFrame, list)
        """

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

        # Empirical Orthogonal Functions
        eofs = pd.DataFrame(pca.components_.transpose(),
                            index=self.anomaly.columns,
                            columns=['eof_' + str(c + 1) for c in range(self.n_components)])

        # Loadings
        pc = self.anomaly.dot(eofs)
        pc.columns = ['pc_' + str(c + 1) for c in range(self.n_components)]

        # Explained variance by each EOF
        explained_variance = list(map(lambda x: x * 100, pca.explained_variance_ratio_))

        # Clear some memory
        del z, s
        gc.collect()

        return eofs, pc, explained_variance

    def project(self, new_data, standardize=False):
        """
        Project data onto the EOFs to obtain its PCs
        """
        # Get new data as anomalies
        if standardize:
            std = new_data.std()
            new_anomaly = (new_data - self.mean) / std
        else:
            new_anomaly = new_data - self.mean

        # Get the PCs
        new_pcs = new_anomaly.to_numpy().dot(self.eof.to_numpy())

        return new_pcs, new_anomaly

    def regression(self, test_data):
        """
        Calculate the PCA regression and the reconstruction error
        """
        pcs, anomaly = self.project(test_data, self.standardize)
        regression = pcs.dot(self.eof.to_numpy().transpose())
        regression = pd.DataFrame(regression, index=test_data.index, columns=self.eof.index)
        residual = regression - anomaly

        return regression, residual, pcs, anomaly


def linear_regression(x: pd.DataFrame, y: pd.DataFrame):
    """
    Get linear regression using pandas and calculate the residuals of the predicted values
    """
    # Clean DataFrames of possible conflictive values
    x = avocado.utils.clean_dataset(x)
    y = avocado.utils.clean_dataset(y)

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



