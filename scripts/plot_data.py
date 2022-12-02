"""
This script represents the observations time series in the desired 
Inputs:
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
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

def open_observations(file_path):
    # Extract the variable name from the path of the file
    variable = (file_path.split('/')[-1]).split('_')[1]
    # Open de file as a dataframe
    data = pd.read_csv(str(file_path), index_col=0)
    # Take only the validated data
    # idx = data.index[data['Validación'] != 'V'].tolist()
    # data = data.drop(['Validación'], axis=1)
    # data.loc[idx] = np.nan

    # Drop column of validated samples, only in the case of statistical variables
    if 'Muestras Validadas' in data.columns:
        data = data.drop(['Muestras Validadas'], axis=1)

    # Rename the columns from latin encoding to avoid compatibility probles
    #data = data.rename({'Media': variable + '_mean',
    #                    'Mínima': variable + '_min',
    #                    'Máxima': variable + '_max',
    #                    'Mediana': variable + '_median',
    #                    'Total': variable + '_total',
    #                    'Tipo': variable + '_type'}, axis='columns')

    # Change format and name of the index to datetime
    data.index = pd.to_datetime(data.index)
    data.index.names = ['dates']

    return data


########################################################################################################################
#                                       REPRESENT THE OBSERVATIONS                                                     #
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
        # Take only the desired period to represent
        period_to_represent = pd.date_range(start=date_ini, end=date_end, freq='H')
        observations = observations.reindex(period_to_represent)
        # Stop the loop
        break

# Plot the variable in the selected period
observations.plot()
plt.show()
