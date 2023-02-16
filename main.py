"""
Main program to do automatic validation of meteorological data

Contact: alvaro@intermet.es
"""

import autoval.validation_tests
import autoval.utils as utils
import matplotlib.pyplot as plt


variables_to_validate = ['TMPA', 'WSPD', 'RADS01', 'PCNR', 'RHMA']

station_to_validate = 'PN001002'

reference_station = 'PN001004'

# Validation parameters
validation_start = [2020, 1, 1, 0, 0]
validation_end = [2020, 12, 31, 23, 59]
sampling_frequency = '1H'

if __name__ == '__main__':

    observations = utils.open_observations('./data/' + station_to_validate + '/', variables_to_validate)
    reference_observations = utils.open_observations('./data/' + reference_station + '/', variables_to_validate)

    observations = observations.AutoVal.impossible_values(
        variables_to_validate,
        start=validation_start,
        end=validation_end,
        freq=sampling_frequency
    )

    observations = observations.AutoVal.climatological_coherence(
        variables_to_validate,
        percentiles=[0.05, 0.95],
        start=validation_start,
        end=validation_end,
        freq=sampling_frequency
    )

    observations = observations.AutoVal.temporal_coherence(
        variables_to_validate,
        percentiles=[0.05, 0.95],
        start=validation_start,
        end=validation_end,
        freq=sampling_frequency
    )

    observations = observations.AutoVal.variance_test(
        variables_to_validate,
        '1D',
        percentiles=[0.05, 0.95],
        start=validation_start,
        end=validation_end,
        freq=sampling_frequency
    )

    observations = observations.AutoVal.spatial_coherence(
        reference_observations,
        variables_to_validate,
        percentiles=[0.05, 0.95],
        start=validation_start,
        end=validation_end,
        freq=sampling_frequency
    )

    utils.Preprocess(observations).clear_low_radiance()
    observations = observations.AutoVal.internal_coherence(
        percentiles=[0.05, 0.95],
        start=validation_start,
        end=validation_end,
        freq=sampling_frequency
    )

    # Plot the results
    observations.AutoVal.vplot(kind='label_type', start=validation_start, end=validation_end, freq=sampling_frequency)
    observations.AutoVal.vplot(kind='label_count', start=validation_start, end=validation_end, freq=sampling_frequency)
    plt.show()
