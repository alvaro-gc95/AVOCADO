"""
Main program to do automatic validation of meteorological data

Contact: alvaro@intermet.es
"""

import autoval.validation_tests
import autoval.utils as utils
import matplotlib.pyplot as plt

# Variables to validate
to_validate = ['TMPA', 'WSPD', 'RADS01', 'PCNR', 'RHMA']

# Station to validate
stat_val = 'PN001002'

# Reference station
stat_ref = 'PN001004'

if __name__ == '__main__':

    # Open all data from a station
    observations = utils.open_observations('./data/' + stat_val + '/', to_validate)

    # Reference station
    reference_observations = utils.open_observations('./data/' + stat_ref + '/', to_validate)

    # utils.Preprocess(observations).wind_components(substitute=True)

    # Validate
    observations = observations.AutoVal.impossible_values(to_validate)
    observations = observations.AutoVal.climatological_coherence(to_validate)
    observations = observations.AutoVal.temporal_coherence(to_validate)
    observations = observations.AutoVal.spatial_coherence(reference_observations, to_validate)

    utils.Preprocess(observations).clear_low_radiance()
    observations = observations.AutoVal.internal_coherence()

    # Plot the results
    observations.AutoVal.vplot(kind='label_type')
    observations.AutoVal.vplot(kind='label_count')
    plt.show()
