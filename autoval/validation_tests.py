"""
AutoVal 0.0.0

Pandas extension to do automatic meteorological data validation

Contact: alvaro@intermet.es
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import autoval.climate
from autoval.climate import Climatology

impossible_thresholds = {
    'TMPA': [-100, 100],
    'RHMA': [0, 100],
    'WSPD': [0, 200],
    'WDIR': [0, 360],
    'PCNR': [0, 1000],
    'RADS01': [0, 2000],
    'RADS02': [0, 2000],
    'RADL01': [0, 2000],
    'RADL02': [0, 2000]
}


@pd.api.extensions.register_dataframe_accessor("AutoVal")
class AutoValidation:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._variables = [c for c in pandas_obj.columns if c.split('_')[-1] not in ['IV', 'CC', 'TC', 'SC']]

    @staticmethod
    def _validate(obj):
        # Verify there is a column with at least one meteorological variable
        if not (set(obj.columns) & set(impossible_thresholds.keys())):
            raise AttributeError("Must have " + ', '.join(impossible_thresholds.keys()))

    def impossible_values(self, variables=None):
        """
        Label values outside an impossible threshold gap.
        """
        # Variables to validate
        if variables is None:
            variables = self._variables

        # Find the dates of the values above the upper impossible limit and below the lower impossible limit
        for variable in variables:

            self._obj[variable + '_' + str(impossible_thresholds[variable][0])] = impossible_thresholds[variable][0]
            self._obj[variable + '_' + str(impossible_thresholds[variable][1])] = impossible_thresholds[variable][1]

            self._obj = label_validation(
                self._obj,
                variables={variable: variable},
                thresholds=impossible_thresholds[variable],
                label='IV'
            )

            self._obj.drop([variable + '_' + str(impossible_thresholds[variable][0])], inplace=True, axis=1)
            self._obj.drop([variable + '_' + str(impossible_thresholds[variable][1])], inplace=True, axis=1)

        return self._obj

    def climatological_coherence(self, variables=None, percentiles=None, skip_labels=True):
        """
        Label values outside extreme climatological percentile values.
        """
        # Variables to validate
        if variables is None:
            variables = self._variables

        # Climatological percentile threshold to label data as suspicious
        if percentiles is None:
            percentiles = [0.01, 0.99]

        # Data to use to calculate the climatology
        if skip_labels:
            train_data = skip_label(self._obj, labels_to_skip=['IV'])
        else:
            train_data = skip_label(self._obj, labels_to_skip=[])

        # Climatology time series
        climatology = Climatology(train_data).daily_cycle(percentiles=percentiles, to_series=True)
        self._obj = pd.concat([self._obj, climatology], axis=1)

        # Compare the observations with the climatology, finding the dates of the values below the minimum percentile
        # or above the maximum percentile
        self._obj = label_validation(
            self._obj,
            variables=dict(zip(variables, variables)),
            thresholds=percentiles,
            label='CC'
        )

        self._obj.drop(climatology.columns, axis=1, inplace=True)

        return self._obj

    def temporal_coherence(self, variables=None, percentiles=None, skip_labels=True):
        """
        Label values with suspicious time evolution (too abrupt or too constant changes)
        """
        # Variables to validate
        if variables is None:
            variables = self._variables

        # Climatological percentile threshold to label data as suspicious
        if percentiles is None:
            percentiles = [0.01, 0.99]

        # Get the variable delta for each time step
        delta_data = self._obj[variables].diff()
        delta_data.columns = [c + '_delta' for c in delta_data.columns]
        self._obj = pd.concat([self._obj, delta_data], axis=1)

        # Data to use to calculate the climatology
        if skip_labels:
            train_data = skip_label(self._obj[delta_data.columns], labels_to_skip=['IV'])
        else:
            train_data = skip_label(self._obj[delta_data.columns], labels_to_skip=[])

        # Climatology time series
        climatology = Climatology(train_data).daily_cycle(percentiles=percentiles, to_series=True)
        self._obj = pd.concat([self._obj, climatology], axis=1)

        # Compare the observations with the climatology, finding the dates of the values below the minimum percentile
        # or above the maximum percentile
        self._obj = label_validation(
            self._obj,
            variables=dict(zip(sorted(delta_data.columns), sorted(variables))),
            thresholds=percentiles,
            label='TC'
        )

        self._obj.drop(climatology.columns, axis=1, inplace=True)
        self._obj.drop(delta_data.columns, axis=1, inplace=True)

        return self._obj

    def spatial_coherence(self, related_site, variables=None, min_corr=0.8, percentiles=None, skip_labels=True):
        """
        Label values outside
        """
        # Variables to validate
        if variables is None:
            variables = self._variables

        # Climatological percentile threshold to label data as suspicious
        if percentiles is None:
            percentiles = [0.01, 0.99]

        # Data to use to calculate the climatology
        if skip_labels:
            train_data = skip_label(self._obj, labels_to_skip=['IV'])
        else:
            train_data = skip_label(self._obj, labels_to_skip=[])

        # Climatology of the regression residuals between observations and the reference data
        residuals_climatology, residuals = get_significant_residuals(train_data, related_site, min_corr, percentiles)
        self._obj = pd.concat([self._obj, residuals.astype('float64'), residuals_climatology], axis=1)

        # Compare the observations with the climatology, finding the dates of the values below the minimum percentile
        # or above the maximum percentile
        self._obj = label_validation(
            self._obj,
            variables=dict(zip(sorted(residuals.columns), sorted(variables))),
            thresholds=percentiles,
            label='SC'
        )

        self._obj.drop(residuals_climatology.columns, axis=1, inplace=True)
        self._obj.drop(residuals.columns, axis=1, inplace=True)

        return self._obj

    def rime_alert(self):
        pass

    def vplot(self, kind=None):

        fig = plt.figure()
        ax = fig.subplots(len(self._variables))

        if kind is None:
            kind = 'label_type'

        if kind == 'label_type':
            for i, variable in enumerate(self._variables):

                # Original data
                self._obj[variable].plot(ax=ax[i], color='grey')

                # Impossible values
                if len(self._obj[variable].loc[self._obj[variable + '_IV'] == 1]) > 0:
                    self._obj[variable].loc[self._obj[variable + '_IV'] == 1].plot(ax=ax[i], marker='o', markersize=2,
                                                                                   color='yellow', linewidth=0)
                # Climatological coherence
                if len(self._obj[variable].loc[self._obj[variable + '_CC'] == 1]) > 0:
                    self._obj[variable].loc[self._obj[variable + '_CC'] == 1].plot(ax=ax[i], marker='o', markersize=2,
                                                                                   color='red', linewidth=0)

                # Temporal coherence
                if len(self._obj[variable].loc[self._obj[variable + '_TC'] == 1]) > 0:
                    self._obj[variable].loc[self._obj[variable + '_TC'] == 1].plot(ax=ax[i], marker='o', markersize=2,
                                                                                   color='green', linewidth=0)

                # Spatial coherence
                if len(self._obj[variable].loc[self._obj[variable + '_SC'] == 1]) > 0:
                    self._obj[variable].loc[self._obj[variable + '_SC'] == 1].plot(ax=ax[i], marker='o', markersize=2,
                                                                                   color='blue', linewidth=0)

                ax[i].set_ylabel(variable)

        elif kind == 'label_count':
            for i, variable in enumerate(self._variables):

                # Original data
                self._obj[variable].plot(ax=ax[i], color='grey')

                # Count the number of labels
                label_columns = [col for col in self._obj.columns if col not in self._variables and variable in col]
                self._obj[variable + '_labels'] = self._obj[label_columns].sum(axis=1)

                # Plot points by changing the color depending on the number of suspect labels
                if len(self._obj[variable].loc[self._obj[variable + '_labels'] == 1]) > 0:
                    self._obj[variable].loc[self._obj[variable + '_labels'] == 1].plot(ax=ax[i], marker='o',
                                                                                       markersize=2,
                                                                                       color='green', linewidth=0)
                if len(self._obj[variable].loc[self._obj[variable + '_labels'] == 2]) > 0:
                    self._obj[variable].loc[self._obj[variable + '_labels'] == 2].plot(ax=ax[i], marker='o',
                                                                                       markersize=2,
                                                                                       color='yellow', linewidth=0)
                if len(self._obj[variable].loc[self._obj[variable + '_labels'] >= 3]) > 0:
                    self._obj[variable].loc[self._obj[variable + '_labels'] >= 3].plot(ax=ax[i], marker='o',
                                                                                       markersize=2,
                                                                                       color='red', linewidth=0)
                # Impossible values
                if len(self._obj[variable].loc[self._obj[variable + '_IV'] == 1]) > 0:
                    self._obj[variable].loc[self._obj[variable + '_IV'] == 1].plot(ax=ax[i], marker='o', markersize=2,
                                                                                   color='black', linewidth=0)

                ax[i].set_ylabel(variable)

                self._obj.drop([variable + '_labels'], axis=1, inplace=True)

        return ax


def label_validation(df: pd.DataFrame, variables: dict, thresholds: (list, tuple), label: str):
    """
    Label a point if it is outside of a threshold gap.
    :param df: DataFrame.
    :param variables: dict. {labeling_variable: variable}.
    :param thresholds: list or tuple. validation gap.
    :param label: str. Name of the label.
    :return df: DataFrame. Original dataframe with the validation column added.
    """

    for labeling_variable, variable in variables.items():

        min_condition = df[labeling_variable] < df[labeling_variable + '_' + str(min(thresholds))]
        max_condition = df[labeling_variable] > df[labeling_variable + '_' + str(max(thresholds))]

        labeled_dates = df[variable].loc[min_condition | max_condition].index

        df[variable + '_' + label] = 0
        df.loc[labeled_dates, variable + '_' + label] = 1

    return df


def skip_label(df: pd.DataFrame, labels_to_skip: (list, tuple)):
    """
    Ignore data where the selected labels = 1.
    """
    columns_to_ignore = [c for c in df.columns if c.split('_')[-1] in labels_to_skip]
    label_columns = [c for c in df.columns if c.split('_')[-1] in ['IV', 'CC', 'SC', 'TC']]

    for col in columns_to_ignore:
        df = df[df[col] != 1]

    df.drop(label_columns, axis=1, inplace=True)

    return df


def get_significant_residuals(original: pd.DataFrame, reference: pd.DataFrame, correlation_threshold, percentiles):

    regression, residuals = Climatology(original).spatial_regression(reference)
    regression_series = autoval.climate.table_to_series(regression, original.index)

    correlation_columns = [c for c in regression_series.columns if 'correlation' in c]

    for col in correlation_columns:
        variable = col.split('_')[0]
        non_significant_dates = regression_series[col].loc[regression_series[col] < correlation_threshold].index
        residuals.loc[non_significant_dates, variable + '_residuals'] = np.nan

    residuals_climatology = Climatology(residuals.astype('float64')).daily_cycle(percentiles=percentiles,
                                                                                 to_series=True)

    return residuals_climatology, residuals
