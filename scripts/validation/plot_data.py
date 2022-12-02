"""
This script represents observations in one or more stations in the desired timeframe
Inputs:
- stations or list of stations separated by commas
- observed variable to plot
- initial date of the timeframe to plot
- ending date of the timeframe to plot
"""

import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
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
stations = args.station.split(',')
observed_variable = args.variable
date_ini = args.DateIni
date_end = args.DateEnd


########################################################################################################################
#                                                 FUNCTIONS                                                            #
########################################################################################################################

# Function to open the file of observations
def open_observations(site, variable):
    # Declare an empty dataframe for the observations
    data = pd.DataFrame()
    # Observations files path
    observations_path = '../data/' + site + '/'
    # List of all the files in the directory of observations of the station
    files = [f for f in os.listdir(observations_path) if os.path.isfile(os.path.join(observations_path, f))]
    print(files)
    # Search the desired observed variable file through all the files in the directory
    for file in files:
        # Open if the file corresponds to the selected variable
        if file.find(variable) != -1:
            # Open the file
            data = pd.read_csv(observations_path + file, index_col=0)
            # Change the format of the index to datetime
            data.index = pd.to_datetime(data.index)
            # Remove the validations columns
            data = data.drop(['impossible values',
                              'climatological coherence',
                              'time step',
                              'spatial coherence'], axis=1)
            # Take only the desired period to represent
            period_to_represent = pd.date_range(start=date_ini, end=date_end, freq='H')
            data = data.reindex(period_to_represent)
            # As the desired variable has now been found, stop the loop
            break
    # Check if the data exists
    if data.empty:
        print('Warning: Empty data. A file for the variable ' + variable + ' may not exist in ' + observations_path)
        exit()

    return data


########################################################################################################################
#                                       REPRESENT THE OBSERVATIONS                                                     #
########################################################################################################################

# Declare the figure
fig = plt.figure()
fig.suptitle(observed_variable)
ax = fig.add_subplot(111)
# Open the files of each of the selected stations separately
for i in range(len(stations)):
    # Station name
    station = stations[i]
    # Open the observations
    observations = open_observations(station, observed_variable)
    # Plot the variable in the selected period
    observations.plot(ax=ax)
ax.legend(stations)
# Adjust the space between the subplots to zero
fig.subplots_adjust(hspace=0)
plt.show()
