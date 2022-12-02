"""
This script check if there are impossible values in the observations of one station in the desired timeframe
Inputs:
- station to validate
- observed variable to validate
- initial date of the timeframe to validate
- ending date of the timeframe to validate
"""

import utils

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


def label_impossible_values(args):
    """
    Label values outside an impossible threshold gap as "impossible values". Export the label to the original file.
    :param args: class. Contains:
       - station: str. Station name.
       - variable: str. Acronym of the variable to validate.
       - DateIni: Datetime. Initial date of the period to validate.
       - DateEnd: Datetime. Final date of the period to validate.
    """
    # Open the observations file
    observations, filename = utils.open_observations('../data/' + args.station + '/',
                                                     args.variable,
                                                     start=args.DateIni,
                                                     end=args.DateEnd,
                                                     freq='H')

    # Find the dates of the values above the upper impossible limit and below the lower impossible limit
    impossible_dates = observations.loc[(observations['value'] < impossible_thresholds[args.variable][0]) |
                                        (observations['value'] > impossible_thresholds[args.variable][1])].index
    # Mark with a "1" the dates of the impossible values, and with a "0" the dates of the possible values
    observations['impossible values'] = 0
    observations.loc[impossible_dates, 'impossible values'] = 1

    # Save the observations with the new validation labels
    observations.to_csv('../data/' + args.station + '/' + filename, header=True)


# Main
if __name__ == '__main__':
    inputs = utils.get_inputs()
    label_impossible_values(inputs)
