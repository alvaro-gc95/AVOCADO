# AutoVal
AutoVal is a python pandas extension of automatic tests for meteorological data validation.

## Dependencies
This extension works in a python 3.8 environment and needs the following python libraries:
- pandas
- numpy
- sklearn
- matplotlib
- itertools

## Data 
The accepted meteorological variables are:

| Variable  | Acronym |
| ------------- | ------------- |
| Temperature | TMPA |
| Relative Humidity | RHMA |
| Wind Speed | WSPD |
| Wind Direction | WDIR |
| Precipitation | PCNR |
| Incoming Shortwave Radiation | RADS01 |
| Outgoing Shortwave Radiation | RADS02 |
| Incoming Longwave Radiation | RALD01 |
| Outgoing Longwave Radiation | RADL02 |

In order to be read by AutoVal, the required format of the original files is the following:

| Date  | Data type |
| ------------- | ------------- |
| 2000-01-01 00:00:00  | 15.0  |
| 2000-01-01 01:00:00  | 13.4  |
| ...  | ...  |


## Validation Tests

**Impossible values**
```
import autoval.utils
import autoval.validation_tests

file_path = './data/station_name/
variables_to_validate = ['TMPA', 'WSDP', 'RADS01]

# Open all data from a station
observations = autoval.utils.open_observations(file_path, variables_to_validate)

# Validate
observations = observations.AutoVal.impossible_values(variables_to_validate)`
```

**Climatological coherence**
```
percentile_thresholds = [0.01, 0.99]

# Validate
observations = observations.AutoVal.climatological_coherence(variables_to_validate, percentile_thresholds)`
```


**Temporal coherence**
```
percentile_thresholds = [0.01, 0.99]

# Validate
observations = observations.AutoVal.temporal_coherence(variables_to_validate, percentile_thresholds)`
```

**Spatial coherence**
```
percentile_thresholds = [0.01, 0.99]

file_path_reference = './data/reference_station_name/

# Open data from a reference station
observations = autoval.utils.open_observations(file_path_reference, variables_to_validate)

# Validate
observations = observations.AutoVal.climatological_coherence(reference_observations, variables_to_validate, percentile_thresholds)`
```


