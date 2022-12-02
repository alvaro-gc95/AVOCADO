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
from sklearn.linear_model import LinearRegression

########################################################################################################################
#                                                  INPUTS                                                              #
########################################################################################################################

# Get the input arguments from the terminal
parser = argparse.ArgumentParser()
parser.add_argument('--reference', type=str, help='Station of reference')
parser.add_argument('--station', type=str, help='Station to compare against the reference station')
parser.add_argument('--variable', type=str, help='Observed variable acronym')
args = parser.parse_args()

# Put the argument objects in variables
reference_station = args.reference
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

    # Change format and name of the index to datetime
    data.index = pd.to_datetime(data.index)
    data.index.names = ['dates']

    return data


def variable_regression(reference_station, station):
    correlation = reference_station.corrwith(station)
    idx = reference_station.index.intersection(station.index)
    reference_station = reference_station.reindex(idx)
    station = station.reindex(idx)
    for col in correlation.dropna().index:
        # Calculate the linear regression for the scatter
        # Create object for the class
        linear_regressor = LinearRegression()
        # Mask for removing nans in the samples
        mask = ~np.isnan(reference_station[col].values) & ~np.isnan(station[col].values)
        # Perform linear regression
        linear_regressor.fit(reference_station.loc[mask, col].values.reshape(-1, 1),
                             station.loc[mask, col].values.reshape(-1, 1))
        # Y predicted value
        y_pred = linear_regressor.predict(reference_station.loc[mask, col].values.reshape(-1, 1))

    return correlation

########################################################################################################################
#                                                                                                                      #
########################################################################################################################

# reference station path
reference_path = '../data/' + reference_station + '/'
# station path
station_path = '../data/' + station + '/'

# Files in the station observed data directory
reference_files = [f for f in os.listdir(reference_path) if os.path.isfile(os.path.join(reference_path, f))]
for file in reference_files:
    # Open if the observation file correspondent to the selected variable
    if file.find(observed_variable) != -1:
        # Open the observed variable file
        reference_observations = open_observations(reference_path + file)
        # Calculate the climatology of the variable
        for col in reference_observations.columns:
            # Use the columns with numeric values
            if reference_observations[col].dtypes != object:
                reference_observations.drop(columns=[col])

# Files in the station observed data directory
station_files = [f for f in os.listdir(station_path) if os.path.isfile(os.path.join(station_path, f))]
for file in station_files:
    # Open if the observation file correspondent to the selected variable
    if file.find(observed_variable) != -1:
        # Open the observed variable file
        station_observations = open_observations(station_path + file)
        # Calculate the climatology of the variable
        for col in station_observations.columns:
            # Use the columns with numeric valuesreference
            if station_observations[col].dtypes != object:
                station_observations.drop(columns=[col])

print(reference_observations)
print(station_observations)
corr = variable_regression(reference_observations, station_observations)
print(corr)