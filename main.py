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
validation_start = [2014, 1, 1, 0, 0]
validation_end = [2014, 12, 31, 23, 59]
sampling_frequency = '1H'

if __name__ == '__main__':

    observations = utils.open_observations('./data/' + station_to_validate + '/', variables_to_validate)
    reference_observations = utils.open_observations('./data/' + reference_station + '/', variables_to_validate)

    observations = observations.validate.impossible_values(
        variables_to_validate,
        start=validation_start,
        end=validation_end,
        freq=sampling_frequency
    )

    observations = observations.validate.climatological_coherence(
        variables_to_validate,
        percentiles=[0.01, 0.99],
        start=validation_start,
        end=validation_end,
        freq=sampling_frequency
    )

    observations = observations.validate.variance_test(
        variables_to_validate,
        '1D',
        percentiles=[0.01, 0.99],
        start=validation_start,
        end=validation_end,
        freq=sampling_frequency
    )

    observations = observations.validate.spatial_coherence(
        reference_observations,
        variables_to_validate,
        min_corr=0.8,
        percentiles=[0.01, 0.99],
        start=validation_start,
        end=validation_end,
        freq=sampling_frequency
    )

    cleared_radiation_observations = utils.Preprocess(observations).clear_low_radiance()
    observations = cleared_radiation_observations.validate.internal_coherence(
        percentiles=[0.01, 0.99],
        start=validation_start,
        end=validation_end,
        freq=sampling_frequency,
        plot_eofs=False,
        plot_reconstruction=True
    )

    # Plot the results
    validation_start_str = ''.join(str(d) for d in validation_start)
    validation_end_str = ''.join(str(d) for d in validation_end)
    plot_name = station_to_validate + '_' + validation_start_str + '_' + validation_end_str

    observations.validate.vplot(kind='label_type', start=validation_start, end=validation_end, freq=sampling_frequency)
    plt.savefig('./' + plot_name + '_type.png')
    plt.close()

    observations.validate.vplot(kind='label_count', start=validation_start, end=validation_end, freq=sampling_frequency)
    plt.savefig('./' + plot_name + '_count.png')
    plt.close()
