"""
AutoVal 0.0.0

Pandas extension to do automatic meteorological data validation

Contact: alvaro@intermet.es
"""

import os
import itertools
import pandas as pd
import matplotlib.pyplot as plt

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

climatological_percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]


@pd.api.extensions.register_dataframe_accessor("AutoVal")
class AutoValidation:
    def __init__(self, pandas_obj):
        self._validate(pandas_obj)
        self._obj = pandas_obj
        self._variables = [c for c in pandas_obj.columns if c.split('_')[-1] not in ['IV', 'CC', 'TC', 'SP']]

    @staticmethod
    def _validate(obj):
        # Verify there is a column with at least one meteorological variable
        if not (set(obj.columns) & set(impossible_thresholds.keys())):
            raise AttributeError("Must have " + ', '.join(impossible_thresholds.keys()))

    def impossible_values(self, variables=None):
        """
        Label values outside an impossible threshold gap.
        """
        if variables is None:
            variables = self._variables
        for variable in variables:
            # Find the dates of the values above the upper impossible limit and below the lower impossible limit
            impossible_dates = self._obj.loc[(self._obj[variable] < impossible_thresholds[variable][0]) |
                                             (self._obj[variable] > impossible_thresholds[variable][1])].index
            # Mark with a "1" the dates of the impossible values, and with a "0" the dates of the possible values
            self._obj[variable + '_IV'] = 0
            self._obj.loc[impossible_dates, variable + '_IV'] = 1
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
        train_data = self._obj.copy()
        label_columns = [c for c in self._obj.columns if c.split('_')[-1] in ['IV', 'CC', 'TC', 'SP']]
        if skip_labels:
            for col in label_columns:
                train_data = train_data[train_data[col] != 1]
        train_data.drop(label_columns, axis=1, inplace=True)

        # Climatology time series
        climatology = Climatology(train_data).daily_cycle(percentiles=percentiles, to_series=True)

        self._obj = pd.concat([self._obj, climatology], axis=1)

        # Compare the observations with the climatology, finding the dates of the values below the minimum percentile
        # or above the maximum percentile
        for variable in variables:
            min_cond = self._obj[variable] < self._obj[variable + '_' + str(min(percentiles))]
            max_cond = self._obj[variable] > self._obj[variable + '_' + str(max(percentiles))]
            anomalous_dates = self._obj[variable].loc[min_cond | max_cond].index

            self._obj[variable + '_CC'] = 0
            self._obj.loc[anomalous_dates, variable + '_CC'] = 1

        self._obj.drop(climatology.columns, axis=1, inplace=True)

        return self._obj

    def spatial_coherence(self):
        """
        Label values outside
        """
        pass

    def temporal_coherence(self):
        pass

    def rime_error(self):
        pass

    # @property
    # def center(self):
    #     # return the geographic center point of this DataFrame
    #     lat = self._obj.latitude
    #     lon = self._obj.longitude
    #     return (float(lon.mean()), float(lat.mean()))
    #
    def vplot(self):

        fig = plt.figure()
        ax = fig.subplots(len(self._variables))

        for i, variable in enumerate(self._variables):

            # Original data
            self._obj[variable].plot(ax=ax[i])

            # Impossible values
            if len(self._obj[variable].loc[self._obj[variable + '_IV'] == 1]) > 0:
                self._obj[variable].loc[self._obj[variable + '_IV'] == 1].plot(ax=ax[i], marker='o', markersize=2,
                                                                               color='yellow', linewidth=0)
            # Impossible values
            elif len(self._obj[variable].loc[self._obj[variable + '_CC'] == 1]) > 0:
                self._obj[variable].loc[self._obj[variable + '_CC'] == 1].plot(ax=ax[i], marker='o', markersize=2,
                                                                               color='red', linewidth=0)

            ax[i].set_ylabel(variable)

        plt.show()
        return ax


class Climatology:
    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def daily_cycle(self, percentiles=None, to_series=False):
        """
        Calculate the percentiles of the monthly daily cycles
        """
        if percentiles is None:
            percentiles = [0.1, 0.25, 0.5, 0.75, 0.9]

        columns = [v + '_' + str(p) for v, p in itertools.product(self._obj.columns, percentiles)]
        # Calculate the index of the monthly daily cycles (format = hour_month)
        idx = [str(h) + '_' + str(m) for h, m in itertools.product(range(0, 24), range(1, 13))]

        # Dataframe of climatological percentiles
        climatology_percentiles = pd.DataFrame(index=idx, columns=columns)
        climatology_series = pd.DataFrame(index=self._obj.index, columns=columns)

        # Calculate the climatological daily cycle for each month for each percentile
        for variable, percentile in itertools.product(self._obj.columns, percentiles):
            for month, month_dataset in self._obj.groupby(self._obj.index.month):
                monthly_climatology = month_dataset[variable].groupby(month_dataset.index.hour).quantile(percentile)
                for hour in monthly_climatology.index:
                    climatology_percentiles.loc[str(hour) + '_' + str(month), variable + '_' + str(percentile)] = \
                        monthly_climatology.loc[hour]

        # transform the monthly daily cycles to time series
        if to_series:
            for variable, percentile in itertools.product(self._obj.columns, percentiles):
                for month, month_dataset in climatology_series.groupby(climatology_series.index.month):
                    for hour, hourly_dataset in month_dataset.groupby(month_dataset.index.hour):
                        climatology_series.loc[hourly_dataset.index, variable + '_' + str(percentile)] = \
                            climatology_percentiles.loc[str(hour) + '_' + str(month), variable + '_' + str(percentile)]
            return climatology_series
        else:
            return climatology_percentiles

    def delta(self):
        pass

    def spatial_corr(self):
        pass


def open_observations(path, variables):
    # Declare an empty dataframe for the complete observations
    data = pd.DataFrame()
    # List of all the files in the directory of observations of the station
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # Search the desired observed variable file through all the files in the directory
    for file, variable in itertools.product(files, variables):
        # Open if the file corresponds to the selected variable
        if file.find(variable) != -1:
            # Open the file
            variable_data = pd.read_csv(path + file, index_col=0)
            # Rename the values column
            variable_data.columns.values[0] = variable
            # Change the format of the index to datetime
            variable_data.index = pd.to_datetime(variable_data.index)
            # Add to the complete DataFrame
            data = pd.concat([data, variable_data], axis=1)
    # Check if the data exists
    if data.empty:
        print('Warning: Empty data. Files may not exist in ' + path)
        exit()
    else:
        return data


if __name__ == '__main__':

    # Variables to validate
    to_validate = ['TMPA', 'WSPD', 'RADS01']

    # Open all data from a station
    observations = open_observations('../data/PN001004/', to_validate)

    # Validate
    observations = observations.AutoVal.impossible_values(to_validate)
    observations = observations.AutoVal.climatological_coherence(to_validate)
    observations.AutoVal.vplot()
