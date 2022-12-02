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
    # idx = data.index[data['Validación'] != 'V'].tolist()
    # data.loc[idx] = np.nan

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


########################################################################################################################
#                                           IMPOSSIBLE VALUES                                                          #
########################################################################################################################
# For temperature (In celsius)
if observed_variable == 'TMPA':
    upper_limit = 100
    lower_limit = -100

# For relative humidity (In %)
if observed_variable == 'RHMA':
    upper_limit = 103
    lower_limit = -1

# For wind speed (In m/s)
if observed_variable == 'WSPD':
    upper_limit = 100
    lower_limit = -1

# For wind direction (In º degrees)
if observed_variable == 'WDIR':
    upper_limit = 361
    lower_limit = -1

# For incoming shortwave radiation (In W/m²)
if observed_variable == 'RADS01':
    upper_limit = 2000
    lower_limit = -1

# For outgoing shortwave radiation (In W/m²)
if observed_variable == 'RADS02':
    upper_limit = 2000
    lower_limit = -1

# For incoming longwave radiation (In W/m²)
if observed_variable == 'RADL01':
    upper_limit = 2000
    lower_limit = -1

# For outgoing longwave radiation (In W/m²)
if observed_variable == 'RADL02':
    upper_limit = 2000
    lower_limit = -1

########################################################################################################################
#                                              FIND IMPOSSIBLE VALUES                                                  #
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
        # Find impossible values in the observations
        for col in observations.columns:
            # Seek only in the columns with numeric values
            if observations[col].dtypes != object:
                impossible_values = observations.loc[(observations[col] > upper_limit) |
                                                     (observations[col] < lower_limit)].index
                # Change the validation label in the original file
                observations.loc[impossible_values, 'Validation'] = 'I'
        # Save the observation with the new validation labels
        observations.to_csv(observations_path + file, header=True)
        # Stop the loop
        break


########################################################################################################################
#                                              FIND IMPOSSIBLE VALUES                                                  #
########################################################################################################################
# Plot the original observations
fig = plt.figure()
ax = plt.subplot(111)
observations.plot(ax=ax)
observations.loc[observations['Validation'] == 'I'].reindex(observations.index).plot(ax=ax, color='red')
plt.show()
