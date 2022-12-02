"""

"""

import yaml
import os
import argparse
import itertools
import pandas as pd
from datetime import datetime


class Configuration:
    """
    Store the configuration parameters
    """

    def __init__(self):
        config = open_yaml('../config.yaml')

        self.stations = config.get('stations')
        self.variables = config.get('variables')
        self.reference = config.get('reference')
        self.DateIni = datetime.strptime(config.get('DateIni'), '%d-%m-%Y')
        self.DateEnd = datetime.strptime(config.get('DateEnd'), '%d-%m-%Y')

    def time(self, freq):
        return pd.date_range(start=self.DateIni, end=self.DateEnd, freq=freq)

    def make_inputs(self):
        inputs = [None]*len(self.stations)*len(self.variables)
        for i, (station, variable) in enumerate(list(itertools.product(self.stations, self.variables))):
            inputs[i] = Inputs(station, self.reference, variable, self.DateIni, self.DateEnd)
        return inputs


class Inputs:
    """
    Store function inputs for individual stations and variables.
    """

    def __init__(self, station, reference, variable, date_ini, date_end):
        self.station = station
        self.reference = reference
        self.variable = variable
        self.DateIni = date_ini
        self.DateEnd = date_end


def open_yaml(yaml_path):
    """
    Read the configuration yaml file.
    :param yaml_path: str. Path of the yaml file
    :return configuration file: Object. Object containing the information of the configuration file.
    """
    # Check if the yaml exists
    if not os.path.exists(yaml_path):
        print(
            'WARNING: The configuration file ' + yaml_path + ' does not exist')
        exit()
    # Read data in ini
    with open(yaml_path, 'r') as stream:
        try:
            configuration_file = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)

    return configuration_file


def get_inputs():
    """
    Get the input arguments from the terminal
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--station', type=str, help='Station of origin of the observed data')
    parser.add_argument('--variable', type=str, help='Observed variable acronym')
    parser.add_argument('--reference', type=str, help='Reference station')
    parser.add_argument('--DateIni', type=lambda d: datetime.strptime(d, '%d-%m-%Y'), help='Start date [DD-MM-YYYY]')
    parser.add_argument('--DateEnd', type=lambda d: datetime.strptime(d, '%d-%m-%Y'), help='End date [DD-MM-YYYY]')

    return parser.parse_args()


def open_observations(path, variable, **kwargs):
    """
    Open the station observations of a selected variable. Get a particular time period is optional.
    :param path: str. Path of the files.
    :param variable: str. Acronym of the variable to get.
    Optional:
        - start: Datetime. Initial date.
        - end: Datetime. Final date.
        - freq: str. Frequency of the data.
    :return data, var_file: DataFrame and list. The obtained observed data and the name of the files that contains it.
    """
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
            data.columns.values[0] = 'value'
            # Change the format of the index to datetime
            data.index = pd.to_datetime(data.index)
            # As the desired variable has now been found, stop the loop
            break
    # Check if the data exists
    if data.empty:
        print('Warning: Empty data. A file for the variable ' + variable + ' may not exist in ' + path)
        exit()

    # Get the data uin the selected period
    if 'start' in kwargs.keys():
        start = kwargs['start']
    else:
        start = data.index[0]

    if 'end' in kwargs.keys():
        end = kwargs['end']
    else:
        end = data.index[-1]

    if 'freq' in kwargs.keys():
        freq = kwargs['freq']
    else:
        freq = '1H'

    data = data.reindex(pd.date_range(start=start, end=end, freq=freq))

    return data, file
