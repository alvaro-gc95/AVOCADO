"""
This script represents the observations time series in the desired
Inputs:
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from datetime import datetime

########################################################################################################################
#                                                  INPUTS                                                              #
########################################################################################################################

# Get the input arguments from the terminal
parser = argparse.ArgumentParser()
parser.add_argument('--station', type=str, help='Station of origin of the observed data')
parser.add_argument('--variable', type=str, help='Observed variable acronym')
args = parser.parse_args()

# Put the argument objects in variables
station = args.station
observed_variable = args.variable


########################################################################################################################
#                                                 FUNCTIONS                                                            #
########################################################################################################################

def open_observations(file_path):
    # Extract the variable name from the path of the file
    variable = (file_path.split('/')[-1]).split('_')[1]
    # Open de file as a dataframe
    data = pd.read_csv(str(file_path), index_col=0)
    # Take only the valid data
    idx = data.index[data['Validation'] == 'I'].tolist()
    data.loc[idx] = np.nan

    # Drop column of validated samples, only in the case of statistical variables
    if 'Validated Samples' in data.columns:
        data = data.drop(['Validated Samples'], axis=1)

    # Rename the columns from latin encoding to avoid compatibility problems
    # data = data.rename({'Media': variable + '_mean',
    #                     'Mínima': variable + '_min',
    #                     'Máxima': variable + '_max',
    #                     'Mediana': variable + '_median',
    #                     'Total': variable + '_total',
    #                     'Tipo': variable + '_type'}, axis='columns')

    # Change format and name of the index to datetime
    data.index = pd.to_datetime(data.index)
    data.index.names = ['dates']

    return data


def climatological_test(observations_data, threshold_interval):
    # Percentile thresholds to flag an observation as suspicious
    up_thr = 1 - (1 - threshold_interval) / 2
    dw_thr = (1 - threshold_interval) / 2

    # Calculate the monthly climatology
    # climatology = observations_data.groupby(observations_data.index.month).mean()

    # Dates of observations above the upper percentile threshold
    anomalies_above_up_thr = []
    # Dates of observations below the lower percentile threshold
    anomalies_below_dw_thr = []
    climatology_series = pd.DataFrame(index=observations_data.index, columns=[str(dw_thr), str(up_thr), 'mean'])
    # Group the data by month
    for month, month_dataset in observations_data.groupby(observations_data.index.month):
        # Calculate the monthly daily cycle climatology
        climatology = month_dataset.groupby(month_dataset.index.hour).mean()
        climatology_up = month_dataset.groupby(month_dataset.index.hour).quantile(up_thr)
        climatology_down = month_dataset.groupby(month_dataset.index.hour).quantile(dw_thr)

        for hour, hourly_dataset in month_dataset.groupby(month_dataset.index.hour):
            climatology_series.loc[hourly_dataset.index, str(dw_thr)] = climatology_down.loc[hour]
            climatology_series.loc[hourly_dataset.index, str(up_thr)] = climatology_up.loc[hour]
            climatology_series.loc[hourly_dataset.index, 'mean'] = climatology.loc[hour]
            # Calculate the anomalies
            hourly_anomaly = hourly_dataset - climatology.loc[hour]
            # Save the dates of observations outside the threshold percentiles that delimit the interval of
            # anomalous observations
            hourly_anomaly_dates_up = hourly_dataset.loc[hourly_anomaly > hourly_anomaly.quantile(up_thr)].index
            hourly_anomaly_dates_dw = hourly_dataset.loc[hourly_anomaly < hourly_anomaly.quantile(dw_thr)].index
            # Add the hour to the date
            hourly_anomaly_dates_up = [datei + pd.to_timedelta(hour, unit='h') for datei in hourly_anomaly_dates_up]
            hourly_anomaly_dates_dw = [datei + pd.to_timedelta(hour, unit='h') for datei in hourly_anomaly_dates_dw]

            anomalies_above_up_thr.extend(hourly_anomaly_dates_up)
            anomalies_below_dw_thr.extend(hourly_anomaly_dates_dw)

        # Calculate the monthly anomalies
        # monthly_anomaly = month_dataset - climatology.loc[month]

        # Save the dates of observations outside the threshold percentiles that delimit the interval of anomalous
        # observations
        # anomalies_above_up_thr.extend(
        #     month_dataset.loc[monthly_anomaly > monthly_anomaly.quantile(up_thr)].index)
        # anomalies_below_dw_thr.extend(
        #     month_dataset.loc[monthly_anomaly < monthly_anomaly.quantile(dw_thr)].index)

    return anomalies_below_dw_thr, anomalies_above_up_thr, climatology_series


########################################################################################################################
#                                          CALCULATE THE CLIMATOLOGY                                                   #
########################################################################################################################

# Observations path
observations_path = '../data/' + station + '/'

# Files in the station observed data directory
files = [f for f in os.listdir(observations_path) if os.path.isfile(os.path.join(observations_path, f))]

for file in files:
    # Open if the observation file correspondent to the selected variable
    if file.find(observed_variable) != -1:
        # Open the observed variable file
        observations = open_observations(observations_path + file)
        # Calculate the climatology of the variable
        for col in observations.columns:
            # Use the columns with numeric values
            if observations[col].dtypes != object:
                clim_dates_below_dwthr, clim_dates_above_upthr, clim_series = climatological_test(observations[col], 0.99)
                # Change the validation label in the original file
                observations.loc[clim_dates_below_dwthr, 'Validation'] = 'C'
                observations.loc[clim_dates_above_upthr, 'Validation'] = 'C'
        # Save the observation with the new validation labels
        observations.to_csv(observations_path + file)
        # Stop the loop
        break

########################################################################################################################
#                                              FIND IMPOSSIBLE VALUES                                                  #
########################################################################################################################
# Plot the original observations
fig = plt.figure()
ax = plt.subplot(111)
observations.plot(ax=ax)
clim_series.plot(ax=ax, color='black')
observations.loc[observations['Validation'] == 'C'].reindex(observations.index).plot(ax=ax, style='o')
plt.show()
