"""
Pandas extension to do automatic meteorological data validation

Contact: alvaro@intermet.es
"""
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

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

n_components = 3


@pd.api.extensions.register_dataframe_accessor("validate")
class AutomaticValidation:
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

    def impossible_values(
            self,
            variables=None,
            start=None,
            end=None,
            freq=None
    ):
        """
        Label values outside an impossible threshold gap.
        """
        # Variables to validate
        if variables is None:
            variables = self._variables

        # Split in training and validation datasets
        validation_dataset, _ = self.split(start, end, freq)

        # Find the dates of the values above the upper impossible limit and below the lower impossible limit
        for variable in variables:

            validation_dataset[variable + '_minimum_threshold'] = impossible_thresholds[variable][0]
            validation_dataset[variable + '_maximum_threshold'] = impossible_thresholds[variable][1]

            validation_dataset = put_validation_label(validation_dataset, variables={variable: variable}, label='IV')

            validation_dataset.drop([variable + '_minimum_threshold'], inplace=True, axis=1)
            validation_dataset.drop([variable + '_maximum_threshold'], inplace=True, axis=1)

            label_columns = list(set(validation_dataset.columns) ^ set(self._obj.columns))
            self._obj[label_columns] = np.nan
            self._obj.loc[validation_dataset.index, label_columns] = validation_dataset[label_columns]

        return self._obj

    def climatological_coherence(
            self,
            variables=None,
            percentiles=None,
            start=None,
            end=None,
            freq=None,
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
        validation_dataset, training_dataset = self.split(start, end, freq)

        # Data to use to calculate the climatology
        if skip_labels:
            training_dataset = skip_label(training_dataset, labels_to_skip=['IV'])
        else:
            training_dataset = skip_label(training_dataset, labels_to_skip=[])

        climatology = get_climatological_thresholds(
            training_dataset,
            dates=validation_dataset.index,
            thresholds=percentiles
        )

        validation_dataset = pd.concat([validation_dataset, climatology], axis=1)

        validation_dataset = put_validation_label(
            validation_dataset,
            variables=dict(zip(variables, variables)),
            label='CC'
        )

        validation_dataset.drop(climatology.columns, axis=1, inplace=True)

        label_columns = list(set(validation_dataset.columns) ^ set(self._obj.columns))
        self._obj[label_columns] = np.nan
        self._obj.loc[validation_dataset.index, label_columns] = validation_dataset[label_columns]

        return self._obj

    def temporal_coherence(
            self,
            variables=None,
            percentiles=None,
            start=None,
            end=None,
            freq=None,
            skip_labels=True
    ):
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
        validation_dataset, training_dataset = self.split(start, end, freq)

        # Data to use to calculate the climatology
        if skip_labels:
            training_dataset = skip_label(
                training_dataset.loc[training_dataset.index, delta_data.columns],
                labels_to_skip=['IV']
            )
        else:
            training_dataset = skip_label(
                training_dataset.loc[training_dataset.index, delta_data.columns],
                labels_to_skip=[]
            )

        # Climatology time series
        climatology = get_climatological_thresholds(
            training_dataset,
            dates=validation_dataset.index,
            thresholds=percentiles
        )
        validation_dataset = pd.concat([validation_dataset, climatology], axis=1)

        # Compare the observations with the climatology, finding the dates of the values below the minimum percentile
        # or above the maximum percentile
        validation_dataset = put_validation_label(
            validation_dataset,
            variables=dict(zip(sorted(delta_data.columns), sorted(variables))),
            label='TC'
        )

        validation_dataset.drop(climatology.columns, axis=1, inplace=True)
        validation_dataset.drop(delta_data.columns, axis=1, inplace=True)
        self._obj.drop(delta_data.columns, axis=1, inplace=True)

        label_columns = list(set(validation_dataset.columns) ^ set(self._obj.columns))
        self._obj[label_columns] = np.nan
        self._obj.loc[validation_dataset.index, label_columns] = validation_dataset[label_columns]

        return self._obj

    def variance_test(
            self,
            variables=None,
            validation_window=None,
            percentiles=None,
            start=None,
            end=None,
            freq=None
    ):
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
        validation_dataset, training_dataset = self.split(start, end, freq)

        # Calculate the variance per month in windows of the same size
        training_variances = training_dataset[variables].resample(validation_window).std()
        variances_to_validate = validation_dataset[variables].resample(validation_window).std()

        # Monthly threshold values of variance
        for variable in variables:

            self._obj[variable + '_VC'] = np.nan
            self._obj.loc[validation_dataset.index, variable + '_VC'] = 0

            for month, month_validation in variances_to_validate[variable].groupby(variances_to_validate.index.month):

                training_month_variances = training_variances.loc[
                    training_variances.index.month == month,
                    variable
                ]
                lower_percentile = training_month_variances.quantile(min(percentiles))
                upper_percentile = training_month_variances.quantile(max(percentiles))

                low_extremes = month_validation > upper_percentile
                high_extremes = month_validation < lower_percentile
                label_idx = month_validation.loc[low_extremes | high_extremes].index

                for idx in label_idx:
                    idx = pd.date_range(start=idx, periods=24, freq='H')
                    self._obj.loc[idx, variable + '_VC'] = 1

        return self._obj

    def spatial_coherence(
            self,
            related_site,
            variables=None,
            min_corr=0.8,
            percentiles=None,
            start=None, end=None, freq=None,
            skip_labels=True,
            plot=False
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
        validation_dataset, training_dataset = self.split(start, end, freq)

        # Data to use to calculate the climatology
        if skip_labels:
            training_dataset = skip_label(training_dataset, labels_to_skip=['IV'])
        else:
            training_dataset = skip_label(training_dataset, labels_to_skip=[])

        # Climatology of the regression residuals between observations and the reference data
        training_regression, training_residuals = get_significant_residuals(training_dataset, related_site, min_corr)
        training_residuals = training_residuals.astype('float64')

        residuals_climatology = get_climatological_thresholds(
            training_residuals,
            dates=validation_dataset.index,
            thresholds=percentiles
        )

        validation_regression, validation_residuals = fit_climatology_regression(
            training_regression,
            validation_dataset[variables]
        )

        # Get the residuals of the data to validate
        validation_dataset = pd.concat([validation_dataset, validation_residuals, residuals_climatology], axis=1)

        if plot:
            maximum_columns = [col for col in residuals_climatology.columns if 'maximum' in col]
            minimum_columns = [col for col in residuals_climatology.columns if 'minimum' in col]
            plot_pca_reconstructions(
                original=validation_residuals,
                reconstruction=validation_residuals,
                maximum_err=residuals_climatology[maximum_columns],
                minimum_err=residuals_climatology[minimum_columns]
            )

        # Compare the observations with the climatology, finding the dates of the values below the minimum percentile
        # or above the maximum percentile
        validation_dataset = put_validation_label(
            validation_dataset,
            variables=dict(zip(sorted(training_residuals.columns), sorted(variables))),
            label='SC'
        )

        validation_dataset.drop(residuals_climatology.columns, axis=1, inplace=True)
        validation_dataset.drop(training_residuals.columns, axis=1, inplace=True)

        label_columns = list(set(validation_dataset.columns) ^ set(self._obj.columns))
        self._obj[label_columns] = np.nan
        self._obj.loc[validation_dataset.index, label_columns] = validation_dataset[label_columns]

        return self._obj

    def internal_coherence(
            self,
            percentiles=None,
            start=None,
            end=None,
            freq=None,
            plot_eofs=False,
            plot_reconstruction=False
    ):
        """
        Find relationships between daily climatological variables and label days that deviates from the expected
        behaviour
        """

        # Climatological percentile threshold to label data as suspicious
        if percentiles is None:
            percentiles = 0.99

        # Split in training and validation datasets
        validation_dataset, training_dataset = self.split(start, end, freq)

        # Get daily variables
        daily_training_dataset = Climatology(training_dataset).climatological_variables()
        daily_validation_dataset = Climatology(validation_dataset).climatological_variables()

        # Empty training period dataframes
        training_regression = pd.DataFrame(
            index=daily_training_dataset.index,
            columns=daily_training_dataset.columns
        )
        training_residuals = pd.DataFrame(
            index=daily_training_dataset.index,
            columns=daily_training_dataset.columns
        )
        training_anomaly = pd.DataFrame(
            index=daily_training_dataset.index,
            columns=daily_training_dataset.columns
        )

        # Empty validation period dataframes
        validation_regression = pd.DataFrame(
            index=daily_validation_dataset.index,
            columns=daily_validation_dataset.columns
        )
        validation_residuals = pd.DataFrame(
            index=daily_validation_dataset.index,
            columns=daily_validation_dataset.columns
        )
        validation_anomaly = pd.DataFrame(
            index=daily_validation_dataset.index,
            columns=daily_validation_dataset.columns
        )

        # Principal Components Analysis of daily variables for each month
        for month, monthly_training_dataset in daily_training_dataset.groupby(daily_training_dataset.index.month):

            # Get EOFs, PCAs and explained variance ratios
            pca = autoval.statistics.DaskPCA(
                monthly_training_dataset,
                n_components=n_components,
                mode='T',
                standardize=True
            )
            if plot_eofs:
                plot_eofs_and_pcs(pca, title='Month ' + str(month).zfill(2) + ' modes of variability')

            # Reconstruct the original time series with the PCA
            monthly_validation_regressor = pca.regression(test_data=daily_validation_dataset)
            month_validation_regression, month_validation_residuals, _, month_validation_anomaly = \
                monthly_validation_regressor

            # Get the errors of the reconstructions during the training period
            monthly_training_regressor = pca.regression(test_data=daily_training_dataset)
            month_training_regression, month_training_residuals, _, month_training_anomaly = \
                monthly_training_regressor

            validation_regression.loc[month_validation_regression.index] = month_validation_regression
            validation_anomaly.loc[month_validation_anomaly.index] = month_validation_anomaly
            validation_residuals.loc[month_validation_residuals.index] = month_validation_residuals

            training_regression.loc[month_training_regression.index] = month_training_regression
            training_anomaly.loc[month_training_anomaly.index] = month_training_anomaly
            training_residuals.loc[month_training_residuals.index] = month_training_residuals

        # Get the error of the reconstruction in hourly resolution
        validation_residuals = validation_residuals.resample('H').ffill()
        training_residuals = training_residuals.resample('H').ffill()

        original_variables = {
            'RADST': 'RADS01',
            'TAMP': 'TMPA',
            'TMEAN': 'TMPA',
            'PTOT': 'PCNR',
            'RHMEAN': 'RHMA',
            'VMEAN': 'WSPD'
        }

        self._obj[[col + '_IC' for col in list(set(original_variables.values()))]] = 0

        minimum_band_columns = [variable + '_minimum_threshold' for variable in daily_training_dataset.columns]
        maximum_band_columns = [variable + '_maximum_threshold' for variable in daily_training_dataset.columns]
        maximum_band = pd.DataFrame(columns=maximum_band_columns, index=daily_validation_dataset.index)
        minimum_band = pd.DataFrame(columns=minimum_band_columns, index=daily_validation_dataset.index)

        # Label values with and error above the maximum percentile threshold
        for variable in daily_training_dataset.columns:
            for month, monthly_error in training_residuals.groupby(training_residuals.index.month):

                monthly_regression_error_validation = validation_residuals.loc[
                    validation_residuals.index.month == month,
                    variable
                ]

                lower_percentile = monthly_error[variable].quantile(min(percentiles))
                upper_percentile = monthly_error[variable].quantile(max(percentiles))

                label_idx = monthly_regression_error_validation.loc[
                    (monthly_regression_error_validation >= upper_percentile) |
                    (monthly_regression_error_validation <= lower_percentile)
                ].index
                label_idx = pd.to_datetime(label_idx)

                maximum_band.loc[maximum_band.index.month == month, variable + '_maximum_threshold'] = upper_percentile
                minimum_band.loc[minimum_band.index.month == month, variable + '_minimum_threshold'] = lower_percentile

                for idx in label_idx:
                    idx = pd.date_range(start=idx, periods=24, freq='H')
                    self._obj.loc[idx, original_variables[variable] + '_IC'] = 1

        maximum_band = maximum_band + validation_anomaly.values
        minimum_band = minimum_band + validation_anomaly.values
        maximum_band = maximum_band.astype('float32')
        minimum_band = minimum_band.astype('float32')

        for variable in validation_regression.columns:
            minimum_threshold = minimum_band[variable + '_minimum_threshold']
            maximum_threshold = maximum_band[variable + '_maximum_threshold']

            label_idx = validation_regression.loc[
                (validation_regression[variable] < minimum_threshold) |
                (validation_regression[variable] > maximum_threshold),
                variable
            ].index

            self._obj.loc[label_idx, original_variables[variable] + '_IC'] = 1

        if plot_reconstruction:

            maximum_band = maximum_band.interpolate(axis=0)
            minimum_band = minimum_band.interpolate(axis=0)

            plot_pca_reconstructions(
                validation_anomaly,
                validation_regression,
                minimum_err=minimum_band,
                maximum_err=maximum_band
            )

        return self._obj

    def vplot(self, kind=None, start=None, end=None, freq=None):

        sns.set_palette('muted')
        sns.set_style("darkgrid")

        # Represent only the validation period
        to_validate, _ = self.split(start, end, freq)
        to_plot = self._obj.loc[to_validate.index]

        plt.rc('text', usetex=True)
        plt.rc('font', family='serif')

        fig = plt.figure()
        axs = fig.subplots(len(self._variables))

        if kind is None:
            kind = 'label_type'

        # Plot of the type of validation label
        if kind == 'label_type':

            fig.suptitle('Coherence label type')

            label_type_colors = {
                'IV': 'black',
                'CC': 'tab:red',
                'SC': 'tab:blue',
                'TC': 'tab:green',
                'VC': 'tab:pink',
                'IC': 'gold'
            }

            for i, variable in enumerate(self._variables):

                if len(self._variables) == 1:
                    ax = axs
                else:
                    ax = axs[i]

                # Original data
                to_plot[variable].plot(ax=ax, color='grey')

                # Validation columns
                label_columns = [col for col in to_plot.columns if col not in self._variables and variable in col]

                for label, color in label_type_colors.items():

                    # Get all the test labels
                    variable_label = [c for c in label_columns if label in c]
                    if len(variable_label) > 0:
                        variable_label = variable_label[0]
                    else:
                        continue

                    # Plot the labeled data
                    labeled_values = to_plot.loc[to_plot[variable_label] == 1, variable]
                    if len(labeled_values) > 0:
                        labeled_values.plot(
                            ax=ax,
                            marker='o',
                            markersize=2,
                            color=color,
                            linewidth=0)

                if i != len(self._variables) - 1:
                    ax.set_xticks([])
                    ax.set_xticklabels([])

                ax.set_ylabel(variable)

            markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in
                       label_type_colors.values()]
            fig.legend(
                markers,
                label_type_colors.keys(),
                numpoints=1,
                loc='upper center',
                ncol=len(label_type_colors),
                bbox_to_anchor=(0.5, 0.95)
            )
            fig.subplots_adjust(hspace=0)

        # Plot of the number of validation labels per value
        elif kind == 'label_count':

            fig.suptitle('Number of labels')

            label_number_colors = {
                1: 'tab:blue',
                2: 'tab:green',
                3: 'gold',
                4: 'tab:red',
                5: 'tab:purple'
            }

            for i, variable in enumerate(self._variables):

                if len(self._variables) == 1:
                    ax = axs
                else:
                    ax = axs[i]

                # Original data
                to_plot[variable].plot(ax=ax, color='grey')

                # Validation columns
                label_columns = [col for col in to_plot.columns if col not in self._variables and variable in col]

                # Count the number of labels
                to_plot[variable + '_labels'] = to_plot[label_columns].sum(axis=1)

                # Plot points by changing the color depending on the number of suspect labels
                for n_labels, color in label_number_colors.items():
                    if len(to_plot.loc[to_plot[variable + '_labels'] == n_labels, variable]) > 0:
                        to_plot.loc[to_plot[variable + '_labels'] == n_labels, variable].plot(
                            ax=ax,
                            marker='o',
                            markersize=2,
                            color=color,
                            linewidth=0
                        )

                # Mark Impossible values
                if variable + '_IV' in to_plot and len(to_plot.loc[to_plot[variable + '_IV'] == 1, variable]) > 0:
                    to_plot.loc[to_plot[variable + '_IV'] == 1, variable].plot(
                        ax=ax,
                        marker='o',
                        markersize=2,
                        color='black',
                        linewidth=0
                    )

                ax.set_ylabel(variable)

                # Delete the label counter from the DataFrame
                to_plot.drop([variable + '_labels'], axis=1, inplace=True)

            markers = [plt.Line2D([0, 0], [0, 0], color=color, marker='o', linestyle='') for color in
                       label_number_colors.values()]
            fig.legend(
                markers,
                label_number_colors.keys(),
                numpoints=1,
                loc='upper center',
                ncol=len(label_number_colors),
                bbox_to_anchor=(0.5, 0.95)
            )
            fig.subplots_adjust(hspace=0)

        return ax


def put_validation_label(data: pd.DataFrame, variables: dict, label: str):
    """
    Label a point if it is outside of a threshold gap.
    :param data: DataFrame.
    :param variables: dict. {labeling_variable: variable}.
    :param label: str. Name of the label.
    :return df: DataFrame. Original dataframe with the validation column added.
    """

    for labeling_variable, variable in variables.items():

        min_condition = data[labeling_variable] < data[labeling_variable + '_minimum_threshold']
        max_condition = data[labeling_variable] > data[labeling_variable + '_maximum_threshold']

        labeled_dates = data.loc[min_condition | max_condition, variable].index

        data[variable + '_' + label] = 0
        data.loc[labeled_dates, variable + '_' + label] = 1

    return data


def get_climatological_thresholds(data: pd.DataFrame, dates, thresholds: (list, tuple, np.array)):
    """
    Get the upper and lower thresholds from the climatology of a variable based on the monthly and hourly percentiles.
    :param data: pd.DataFrame.
    :param dates: list, DatetimeIndex. dates in which return the climatology as a time series.
    :param thresholds: list, tuple, array. Percentile thresholds.
    """

    try:
        minimum_threshold = min(thresholds)
        maximum_threshold = max(thresholds)
    except AttributeError:
        raise " thresholds arguments should have at least len=2"

    climatology = Climatology(data).daily_cycle(percentiles=thresholds, to_series=True, dates=dates)

    for col in climatology:
        if str(maximum_threshold) in col:
            climatology = climatology.rename(columns={col: col.replace(str(maximum_threshold), 'maximum_threshold')})
        if str(minimum_threshold) in col:
            climatology = climatology.rename(columns={col: col.replace(str(minimum_threshold), 'minimum_threshold')})

    return climatology


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


def fit_climatology_regression(regressor: pd.DataFrame, x: pd.DataFrame):
    """
    Fit data to a linear regression by month and hour.
    :param regressor: pd.DataFrame. Data of the coefficients, intercepts and correlation coefficients by hour and month.
    :param x: pd.DataFrame. Data to fit.
    """

    y = x.copy()

    for variable, date in itertools.product(x, regressor.index):

        date_hour = int(date.split('_')[0])
        date_month = int(date.split('_')[1])

        dates = (x.index.month == date_month) & (x.index.hour == date_hour)

        coef = regressor.loc[date, variable + '_coef']
        intercept = regressor.loc[date, variable + '_intercept']

        y.loc[dates, variable] = x.loc[dates, variable].apply(lambda _x: _x*coef + intercept)

    residuals = x - y
    residuals.columns = [col + '_residuals' for col in residuals.columns]

    return y, residuals


def get_significant_residuals(original: pd.DataFrame, reference: pd.DataFrame, correlation_threshold: float):
    """
    Get the residuals only when the regression have a correlation coefficient above the selected threshold
    """

    regression, residuals = Climatology(original).spatial_regression(reference)
    regression_series = autoval.climate.table_to_series(regression, original.index)

    correlation_columns = [c for c in regression_series.columns if 'correlation' in c]

    for col in correlation_columns:
        variable = col.split('_')[0]
        non_significant_dates = regression_series.loc[regression_series[col] < correlation_threshold, col].index
        residuals.loc[non_significant_dates, variable + '_residuals'] = np.nan

    return regression, residuals


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


def plot_eofs_and_pcs(pca, title):
    """
    Represent the principal modes of variability for the internal coherence test
    """

    sns.set_palette('muted')
    sns.set_style("darkgrid")
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Get Principal Components and Empirical Orthogonal Functions
    pcs = pca.pc
    eofs = pca.eof

    # Reorganize the EOFs dataframe
    eofs_line = []
    eofs = eofs.reset_index()
    eofs.columns = [col.replace('_', ' ') if col != eofs.columns[0] else 'variable' for col in eofs.columns]

    for col in eofs:
        if col != 'variable':
            individual_eof = eofs[col].to_frame()
            individual_eof['eof'] = col
            individual_eof.columns = ['standardized anomaly', 'eof']
            individual_eof = pd.concat([individual_eof, eofs['variable']], axis=1)

            eofs_line.append(individual_eof)

    eofs_line = pd.concat(eofs_line, axis=0)
    eofs_line = eofs_line.reset_index()

    variables = list(set(eofs_line['variable'].values))

    # Represent the modes of variability
    fig = plt.figure(figsize=(10, 2.5 * pca.n_components))
    fig.suptitle(title)
    ax = fig.subplots(pca.n_components, 2, gridspec_kw={'width_ratios': [1, 3]})
    pcs = pcs.dropna(axis=0)

    for i, col in enumerate(pcs.columns):
        explained_variance = round(pca.explained_variance[i], 2)

        eof = eofs_line.loc[eofs_line['eof'] == 'eof ' + str(i + 1)]
        sns.barplot(data=eof, y='standardized anomaly', x='eof', hue='variable', ax=ax[i, 0])
        handles, labels = ax[i, 0].get_legend_handles_labels()
        ax[i, 0].legend([], [], frameon=False)
        # sns.move_legend(ax[i, 0], "upper left", bbox_to_anchor=(1, 1))
        ax[i, 0].set_ylabel('EOF ' + str(i + 1) + ' (exp. var = ' + str(explained_variance) + '\%)')
        ax[i, 0].set_xticks([])
        ax[i, 0].set_xticklabels([])
        ax[i, 0].set_xlabel('Variables')

        ax[i, 1].axhline(0, color='darkgrey')
        ax[i, 1].plot(pcs[col].values)
        ax[i, 1].set_ylabel('PC ' + str(i + 1))

    fig.legend(
        handles, labels,
        loc='upper center',
        ncol=len(variables),
        bbox_to_anchor=(0.5, 0.95)
    )

    plt.subplots_adjust(hspace=0)
    plt.show()


def plot_pca_reconstructions(original, reconstruction, minimum_err=None, maximum_err=None):
    """
    Represent data and its reconstruction by PCA method
    """

    sns.set_palette('muted')
    sns.set_style("darkgrid")
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')

    # Take only the reconstructed period
    original = original.dropna(axis=0)
    reconstruction = reconstruction.dropna(axis=0)

    fig = plt.figure()
    fig.suptitle('Standarized anomalies')
    ax = fig.subplots(len(reconstruction.columns))

    for i, col in enumerate(reconstruction):
        original[col].plot(ax=ax[i], color='grey', label='original')

        if minimum_err is not None:
            minimum_err[col + '_minimum_threshold'].plot(ax=ax[i], color='tab:red', alpha=0.4)
        if maximum_err is not None:
            maximum_err[col + '_maximum_threshold'].plot(ax=ax[i], color='tab:red', alpha=0.4)
        if maximum_err is not None and minimum_err is not None:
            ax[i].fill_between(
                maximum_err[col + '_maximum_threshold'].index,
                minimum_err[col + '_minimum_threshold'],
                maximum_err[col + '_maximum_threshold'],
                alpha=0.4, color='tab:red'
            )

        reconstruction[col].plot(ax=ax[i], color='tab:blue', alpha=0.6, label='reconstruction')

        ax[i].set_ylabel(col)

    fig.subplots_adjust(hspace=0)

    plt.show()
