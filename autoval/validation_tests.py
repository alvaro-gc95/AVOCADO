"""
Pandas extension to do automatic meteorological data validation

Contact: alvaro@intermet.es
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import autoval.climate
from autoval.climate import Climatology
import autoval.statistics
import autoval.utils

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
        self._variables = [c for c in pandas_obj.columns if c.split('_')[-1] not in ['IV', 'CC', 'TC', 'SC', 'VC']]

    @staticmethod
    def _validate(obj):
        # Verify there is a column with at least one meteorological variable
        if not (set(obj.columns) & set(impossible_thresholds.keys())):
            raise AttributeError("Must have " + ', '.join(impossible_thresholds.keys()))

    def split(self, start, end, freq):
        validation_dataset, training_dataset, = split_data(
            self._obj,
            validation_start=start,
            validation_end=end,
            freq=freq
        )
        return validation_dataset, training_dataset

    def impossible_values(self, variables=None, start=None, end=None, freq=None):
        """
        Label values outside an impossible threshold gap.
        """
        # Variables to validate
        if variables is None:
            variables = self._variables

        # Split in training and validation datasets
        to_validate, _ = self.split(start, end, freq)

        # Find the dates of the values above the upper impossible limit and below the lower impossible limit
        for variable in variables:
            to_validate[variable + '_' + str(impossible_thresholds[variable][0])] = impossible_thresholds[variable][0]
            to_validate[variable + '_' + str(impossible_thresholds[variable][1])] = impossible_thresholds[variable][1]

            to_validate = label_validation(
                to_validate,
                variables={variable: variable},
                thresholds=impossible_thresholds[variable],
                label='IV'
            )

            to_validate.drop([variable + '_' + str(impossible_thresholds[variable][0])], inplace=True, axis=1)
            to_validate.drop([variable + '_' + str(impossible_thresholds[variable][1])], inplace=True, axis=1)

            label_columns = list(set(to_validate.columns) ^ set(self._obj.columns))
            self._obj[label_columns] = np.nan
            self._obj.loc[to_validate.index, label_columns] = to_validate[label_columns].copy()

        return self._obj

    def climatological_coherence(
            self,
            variables=None,
            percentiles=None,
            start=None, end=None, freq=None,
            skip_labels=True
    ):
        """
        Label values outside extreme climatological percentile values.
        """
        # Variables to validate
        if variables is None:
            variables = self._variables

        # Climatological percentile threshold to label data as suspicious
        if percentiles is None:
            percentiles = [0.01, 0.99]

        # Split in training and validation datasets
        to_validate, training_dataset = self.split(start, end, freq)

        # Data to use to calculate the climatology
        if skip_labels:
            training_dataset = skip_label(training_dataset, labels_to_skip=['IV'])
        else:
            training_dataset = skip_label(training_dataset, labels_to_skip=[])

        # Climatology time series
        climatology = Climatology(training_dataset).daily_cycle(
            percentiles=percentiles,
            to_series=True,
            dates=to_validate.index
        )
        to_validate = pd.concat([to_validate, climatology], axis=1)

        # Compare the observations with the climatology, finding the dates of the values below the minimum percentile
        # or above the maximum percentile
        to_validate = label_validation(
            to_validate,
            variables=dict(zip(variables, variables)),
            thresholds=percentiles,
            label='CC'
        )

        to_validate.drop(climatology.columns, axis=1, inplace=True)

        label_columns = list(set(to_validate.columns) ^ set(self._obj.columns))
        self._obj[label_columns] = np.nan
        self._obj.loc[to_validate.index, label_columns] = to_validate[label_columns].copy()

        return self._obj

    def temporal_coherence(self, variables=None, percentiles=None, start=None, end=None, freq=None, skip_labels=True):
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

        # Split in training and validation datasets
        to_validate, training_dataset = self.split(start, end, freq)

        # Data to use to calculate the climatology
        if skip_labels:
            training_dataset = skip_label(
                training_dataset[delta_data.columns].loc[training_dataset.index],
                labels_to_skip=['IV']
            )
        else:
            training_dataset = skip_label(
                training_dataset[delta_data.columns].loc[training_dataset.index],
                labels_to_skip=[]
            )

        # Climatology time series
        climatology = Climatology(training_dataset).daily_cycle(
            percentiles=percentiles,
            to_series=True,
            dates=to_validate.index
        )
        to_validate = pd.concat([to_validate, climatology], axis=1)

        # Compare the observations with the climatology, finding the dates of the values below the minimum percentile
        # or above the maximum percentile
        to_validate = label_validation(
            to_validate,
            variables=dict(zip(sorted(delta_data.columns), sorted(variables))),
            thresholds=percentiles,
            label='TC'
        )

        to_validate.drop(climatology.columns, axis=1, inplace=True)
        to_validate.drop(delta_data.columns, axis=1, inplace=True)
        self._obj.drop(delta_data.columns, axis=1, inplace=True)

        label_columns = list(set(to_validate.columns) ^ set(self._obj.columns))
        self._obj[label_columns] = np.nan
        self._obj.loc[to_validate.index, label_columns] = to_validate[label_columns].copy()

        return self._obj

    def spatial_coherence(
            self,
            related_site,
            variables=None,
            min_corr=0.8,
            percentiles=None,
            start=None, end=None, freq=None,
            skip_labels=True
    ):
        """
        Calculate the regression with a highly correlated station and label values with residuals outside the selected
        percentile gap
        """
        # Variables to validate
        if variables is None:
            variables = self._variables

        # Climatological percentile threshold to label data as suspicious
        if percentiles is None:
            percentiles = [0.01, 0.99]

        # Split in training and validation datasets
        to_validate, training_dataset = self.split(start, end, freq)

        # Data to use to calculate the climatology
        if skip_labels:
            training_dataset = skip_label(training_dataset, labels_to_skip=['IV'])
        else:
            training_dataset = skip_label(training_dataset, labels_to_skip=[])

        # Climatology of the regression residuals between observations and the reference data
        residuals = get_significant_residuals(training_dataset, related_site, min_corr)
        residuals_climatology = Climatology(residuals.astype('float64')).daily_cycle(
            percentiles=percentiles,
            to_series=True,
            dates=to_validate.index
        )

        # Get the residuals of the data to validate
        to_validate = pd.concat([to_validate, residuals.astype('float64'), residuals_climatology], axis=1)

        # Compare the observations with the climatology, finding the dates of the values below the minimum percentile
        # or above the maximum percentile
        to_validate = label_validation(
            to_validate,
            variables=dict(zip(sorted(residuals.columns), sorted(variables))),
            thresholds=percentiles,
            label='SC'
        )

        to_validate.drop(residuals_climatology.columns, axis=1, inplace=True)
        to_validate.drop(residuals.columns, axis=1, inplace=True)

        label_columns = list(set(to_validate.columns) ^ set(self._obj.columns))
        self._obj[label_columns] = np.nan
        self._obj.loc[to_validate.index, label_columns] = to_validate[label_columns].copy()

        return self._obj

    def internal_coherence(self, percentiles=None, start=None, end=None, freq=None):
        """
        Find relationships between daily climatological variables and label days that deviates from the expected
        behaviour
        """

        # Climatological percentile threshold to label data as suspicious
        if percentiles is None:
            percentiles = 0.99

        # Split in training and validation datasets
        to_validate, training_dataset = self.split(start, end, freq)

        # Get daily variables
        daily_training_dataset = Climatology(training_dataset).climatological_variables()
        daily_to_validate = Climatology(to_validate).climatological_variables()

        # Make empty dataframes
        regression = pd.DataFrame(
            index=daily_training_dataset.index,
            columns=daily_training_dataset.columns
        )

        regression_error = pd.DataFrame(
            index=daily_training_dataset.index,
            columns=daily_training_dataset.columns
        )

        anomaly = pd.DataFrame(
            index=daily_training_dataset.index,
            columns=daily_training_dataset.columns
        )

        # Principal Components Analysis of daily variables for each month
        for month, monthly_df in daily_training_dataset.groupby(daily_training_dataset.index.month):
            # Get EOFs, PCAs and explained variance ratios
            pca = autoval.statistics.DaskPCA(monthly_df, n_components=3, mode='T', standardize=True)

            # pcs = pca.pc
            # fig = plt.figure()
            # ax = fig.subplots(3)
            # pcs = pcs.dropna(axis=0)
            # for i, col in enumerate(pcs.columns):
            #     ax[i].plot(pcs[col].values)
            #     ax[i].set_ylabel('PC' + str(i + 1))
            # plt.subplots_adjust(hspace=0)
            # plt.show()
            # import seaborn as sns
            # eofs = pca.eof
            # eofs_line = []
            # eofs = eofs.reset_index()
            # eofs.columns = [col.replace('_', ' ') if col != eofs.columns[0] else 'variable' for col in eofs.columns]
            # for col in eofs:
            #     if col != 'variable':
            #         individual_eof = eofs[col].to_frame()
            #         individual_eof['eof'] = col
            #         individual_eof.columns = ['standardized anomaly', 'eof']
            #         individual_eof = pd.concat([individual_eof, eofs['variable']], axis=1)
            #
            #         eofs_line.append(individual_eof)
            # eofs_line = pd.concat(eofs_line, axis=0)
            # eofs_line = eofs_line.reset_index()
            # sns.set_palette('muted')
            # sns.set_style("darkgrid")
            # ax = sns.barplot(data=eofs_line, y='standardized anomaly', x='eof', hue='variable')
            # plt.title(month)
            # sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
            # plt.show()

            # Reconstruct the original time series with the PCA

            regression_month, regression_error_month, _, anomaly_month = pca.regression(test_data=daily_to_validate)

            regression.loc[regression_month.index] = regression_month
            anomaly.loc[daily_to_validate.index] = anomaly_month
            regression_error.loc[regression_error_month.index] = regression_error

        fig = plt.figure()
        ax = fig.subplots(len(regression.columns))
        for i, col in enumerate(regression):
            anomaly[col].plot(ax=ax[i], color='grey')
            regression[col].plot(ax=ax[i], color='tab:blue', alpha=0.6)
        plt.show()

        # Get the error of the reconstruction in hourly resolution
        regression_error = anomaly - regression
        # regression_error = regression_error.where(regression_error > 0, np.nan)
        regression_error = regression_error.resample('H').ffill()

        original_variables = {
            'RADST': 'RADS01',
            'TAMP': 'TMPA',
            'TMEAN': 'TMPA',
            'PTOT': 'PCNR',
            'RHMEAN': 'RHMA',
            'VMEAN': 'WSPD'
        }

        self._obj[[col + '_IC' for col in list(set(original_variables.values()))]] = 0

        # Label values with and error above the maximum percentile threshold
        for variable in daily_training_dataset.columns:

            lower_percentile = regression_error[variable].quantile(min(percentiles))
            upper_percentile = regression_error[variable].quantile(max(percentiles))

            label_idx = regression_error[variable].loc[
                    (regression_error[variable] >= upper_percentile) |
                    (regression_error[variable] <= lower_percentile)
                    ].index
            label_idx = pd.to_datetime(label_idx)

            for idx in label_idx:
                idx = pd.date_range(start=idx, periods=24, freq='H')
                self._obj[original_variables[variable] + '_IC'].loc[idx] = 1

        return self._obj

    def variance_test(self, variables=None, validation_window=None, percentiles=None, start=None, end=None, freq=None):
        """
        Label values within a selected window of time with an anomalous variance (too high or too low)
        """

        # Variables to validate
        if variables is None:
            variables = self._variables

        # Validation window size
        if validation_window is None:
            validation_window = '1D'

        # Percentile threshold
        if percentiles is None:
            percentiles = [0.01, 0.99]

        # Split in training and validation datasets
        to_validate, training_dataset = self.split(start, end, freq)

        # Calculate the variance per month in windows of the same size
        training_variances = training_dataset[variables].resample(validation_window).std()
        variances_to_validate = to_validate[variables].resample(validation_window).std()

        # Monthly threshold values of variance
        for variable in variables:

            self._obj[variable + '_VC'] = np.nan
            self._obj[variable + '_VC'].loc[to_validate.index] = 0

            for month, month_dataset in variances_to_validate[variable].groupby(variances_to_validate.index.month):

                monthly_variances = training_variances[variable].loc[training_variances.index.month == month]
                lower_percentile = monthly_variances.quantile(min(percentiles))
                upper_percentile = monthly_variances.quantile(max(percentiles))

                low_extremes = month_dataset >= upper_percentile
                high_extremes = month_dataset <= lower_percentile
                label_idx = month_dataset.loc[low_extremes | high_extremes].index

                for idx in label_idx:
                    idx = pd.date_range(start=idx, periods=24, freq='H')
                    self._obj[variable + '_VC'].loc[idx] = 1

        return self._obj

    def vplot(self, kind=None):

        fig = plt.figure()
        axs = fig.subplots(len(self._variables))

        if kind is None:
            kind = 'label_type'

        # Plot of the type of validation label
        if kind == 'label_type':

            fig.suptitle('LABEL TYPE')

            label_type_colors = {
                'IV': 'black',
                'CC': 'red',
                'SC': 'blue',
                'TC': 'green',
                'VC': 'pink',
                'IC': 'yellow'
            }

            for i, variable in enumerate(self._variables):

                if len(self._variables) == 1:
                    ax = axs
                else:
                    ax = axs[i]

                # Original data
                self._obj[variable].plot(ax=ax, color='grey')

                # Validation columns
                label_columns = [col for col in self._obj.columns if col not in self._variables and variable in col]

                for label, color in label_type_colors.items():

                    # Get all the test labels
                    variable_label = [c for c in label_columns if label in c]
                    if len(variable_label) > 0:
                        variable_label = variable_label[0]
                    else:
                        continue

                    # Plot the labeled data
                    if len(self._obj[variable].loc[self._obj[variable_label] == 1]) > 0:
                        self._obj[variable].loc[self._obj[variable_label] == 1].plot(
                            ax=ax,
                            marker='o',
                            markersize=2,
                            color=color,
                            linewidth=0)

                ax.set_ylabel(variable)

            markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in
                       label_type_colors.values()]
            plt.legend(markers, label_type_colors.keys(), numpoints=1)

        # Plot of the number of validation labels per value
        elif kind == 'label_count':

            fig.suptitle('LABEL COUNT')

            label_number_colors = {
                1: 'blue',
                2: 'green',
                3: 'yellow',
                4: 'red',
                5: 'purple'
            }

            for i, variable in enumerate(self._variables):

                if len(self._variables) == 1:
                    ax = axs
                else:
                    ax = axs[i]

                # Original data
                self._obj[variable].plot(ax=ax, color='grey')

                # Validation columns
                label_columns = [col for col in self._obj.columns if col not in self._variables and variable in col]

                # Count the number of labels
                self._obj[variable + '_labels'] = self._obj[label_columns].sum(axis=1)

                # Plot points by changing the color depending on the number of suspect labels
                for n_labels, color in label_number_colors.items():
                    if len(self._obj[variable].loc[self._obj[variable + '_labels'] == n_labels]) > 0:
                        self._obj[variable].loc[self._obj[variable + '_labels'] == n_labels].plot(
                            ax=ax,
                            marker='o',
                            markersize=2,
                            color=color,
                            linewidth=0
                        )

                # Mark Impossible values
                if (variable + '_IV' in self._obj) and \
                        (len(self._obj[variable].loc[self._obj[variable + '_IV'] == 1]) > 0):
                    self._obj[variable].loc[self._obj[variable + '_IV'] == 1].plot(
                        ax=ax,
                        marker='o',
                        markersize=2,
                        color='black',
                        linewidth=0
                    )

                ax.set_ylabel(variable)

                # Delete the label counter from the DataFrame
                self._obj.drop([variable + '_labels'], axis=1, inplace=True)

            markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in
                       label_number_colors.values()]
            plt.legend(markers, label_number_colors.keys(), numpoints=1)

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
    label_columns = [c for c in df.columns if c.split('_')[-1] in ['IV', 'CC', 'SC', 'TC', 'VC']]

    for col in columns_to_ignore:
        df = df[df[col] != 1]

    df.drop(label_columns, axis=1, inplace=True)

    return df


def get_significant_residuals(original: pd.DataFrame, reference: pd.DataFrame, correlation_threshold: float):
    """
    Get the residuals only when the regression have a correlation coefficient above the selected threshold
    """

    regression, residuals = Climatology(original).spatial_regression(reference)
    regression_series = autoval.climate.table_to_series(regression, original.index)

    correlation_columns = [c for c in regression_series.columns if 'correlation' in c]

    for col in correlation_columns:
        variable = col.split('_')[0]
        non_significant_dates = regression_series[col].loc[regression_series[col] < correlation_threshold].index
        residuals.loc[non_significant_dates, variable + '_residuals'] = np.nan

    return residuals


def split_data(data: (pd.DataFrame, pd.Series), validation_start: list, validation_end: list, freq='1H'):
    """
    Split the data int he training dataset and validation dataset.
    :param data: pd.DataFrame or pd.Series. Original dataset.
    :param validation_start: list. Starting date of the validation period.
    :param validation_end: list. Ending date of the validation period.
    :param freq: str. Sampling frequency of the validation period.
    """
    validation_start = datetime(*validation_start)
    validation_end = datetime(*validation_end)
    validation_period = pd.date_range(start=validation_start, end=validation_end, freq=freq)

    # Check that the validation period is in the original dataset
    validation_period = sorted(list(set(validation_period) & set(data.index)))

    validation_dataset = data.loc[validation_period]
    training_dataset = data.drop(validation_period, axis=0)

    return validation_dataset, training_dataset
