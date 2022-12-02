"""
This script check if there are values above/below a maximum/minimum climatological percentile. This values are extreme
in comparison with the climatology and therefore climatologically incoherent.
Inputs:
- station to validate
- observed variable to validate
- initial date of the timeframe to validate
- ending date of the timeframe to validate
"""

import os
import pandas as pd
import argparse
from datetime import datetime

########################################################################################################################
#                                                  INPUTS                                                              #
########################################################################################################################

# Get the input arguments from the terminal
parser = argparse.ArgumentParser()
parser.add_argument('--station', type=str, help='Station of origin of the observed data')
parser.add_argument('--variable', type=str, help='Observed variable acronym')
parser.add_argument('--DateIni', type=lambda d: datetime.strptime(d, '%d-%m-%Y'), help='Start date [DD-MM-YYYY]')
parser.add_argument('--DateEnd', type=lambda d: datetime.strptime(d, '%d-%m-%Y'), help='End date [DD-MM-YYYY]')
args = parser.parse_args()

# Put the argument objects in variables
station = args.station
observed_variable = args.variable
date_ini = args.DateIni
date_end = args.DateEnd


########################################################################################################################
#                                                 FUNCTIONS                                                            #
########################################################################################################################

# Function to open the file of observations
def open_observations(path, variable):
    # Declare an empty dataframe for the observations
    data = pd.DataFrame()
    # List of all the files in the directory of observations of the station
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # Search the desired observed variable file through all the files in the directory
    for file in files:
        # Open if the file corresponds to the selected variable
        if file.find(variable) != -1:
            # Open the file
            data = pd.read_csv(path + file, index_col=0)
            # Change the format of the index to datetime
            data.index = pd.to_datetime(data.index)
            # As the desired variable has now been found, stop the loop
            break
    # Check if the data exists
    if data.empty:
        print('Warning: Empty data. A file for the variable ' + variable + ' may not exist in ' + path)
        exit()
    return data, file


# Function to open the climatology file
def open_climatology(path, variable):
    # Declare an empty dataframe for the observations
    climatology_data = pd.DataFrame()
    # List of all the files in the directory of observations of the station
    files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    # Search the desired observed variable file through all the files in the directory
    for file in files:
        # Open if the file corresponds to the selected variable
        if file.find(variable) != -1:
            # Open the file
            climatology_data = pd.read_csv(path + file, index_col=0)
            # As the desired variable has now been found, stop the loop
            break
    # Check if the data exists
    if climatology_data.empty:
        print('Warning: Empty data. A climatology file for the variable ' + variable + ' may not exist in ' + path)
        exit()

    return climatology_data


# Function to check the climatological coherence
def climatological_test(observations_data, climatology_data, percentiles, period_to_test):
    # Select the period to test
    observations_to_test = observations_data.reindex(period_to_test).drop(['impossible values',
                                                                           'climatological coherence',
                                                                           'time step',
                                                                           'spatial coherence'], axis=1)
    # Transform the climatology table into a time series
    climatology_series = pd.DataFrame(index=period_to_test, columns=[str(p) for p in percentiles])
    for month, month_dataset in climatology_series.groupby(climatology_series.index.month):
        for hour, hourly_dataset in month_dataset.groupby(month_dataset.index.hour):
            for percentile in percentiles:
                climatology_series.loc[hourly_dataset.index, str(percentile)] = \
                    climatology_data.loc[str(hour) + '_' + str(month), 'value_' + str(percentile)]
    # Compare the observations with the climatology, finding the dates of the values below the minimum percentile or
    # above the maximum percentile
    observations_to_test = observations_to_test.join(climatology_series)
    anomalous_dates = observations_to_test.loc[(observations_to_test['value'] <
                                                observations_to_test[str(min(percentiles))]) |
                                               (observations_to_test['value'] >
                                                observations_to_test[str(max(percentiles))])].index

    return anomalous_dates


########################################################################################################################
#                                    FIND THE CLIMATOLOGICALLY INCOHERENT OBSERVATIONS                                 #
########################################################################################################################

# Observations path
observations_path = '../data/' + station + '/'
# List of percentiles
percentiles = [0.01, 0.99]
# Timeframe of the observations to test
timeframe = pd.date_range(start=date_ini, end=date_end, freq='H')
# Open the observations
observations, filename = open_observations(observations_path, observed_variable)
# Open the climatology
climatology = open_climatology(observations_path + 'climatology/', observed_variable)
# Test the climatological coherence in the selected timeframe
climatological_anomalies = climatological_test(observations, climatology, percentiles, timeframe)
# Label the dates obtained with a "1" if they are suspicious of climatological incoherence, and with a "0" if not
observations.loc[climatological_anomalies, 'climatological coherence'] = 1
observations.loc[timeframe.drop(climatological_anomalies), 'climatological coherence'] = 0

# Save the observations with the new validation labels
observations.to_csv(observations_path + filename, header=True)

