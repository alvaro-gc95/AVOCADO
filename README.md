# AutoVal
AutoVal is a python pandas extension of automatic tests for meteorological data validation.

# Dependencies
This extension works in a python 3.8 environment and needs the following python libraries:
- pandas
- numpy
- sklearn
- matplotlib
- itertools

# Data 
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

# Impossible values
```
import autoval.utils
import autoval.validation_tests


# Open all data from a station
observations = autoval.utils.open_observations('./data/' + stat_val + '/', to_validate)

# Validate
observations = observations.AutoVal.impossible_values(to_validate)`

```


- Impossible values: pd.DataFrame.AutoVal.impossible_values()
- Climatological coherence: pd.DataFrame.AutoVal.climatological_coherence()
- Temporal coherence: pd.DataFrame.AutoVal.temporal_coherence()
- Spatial coherence: pd.DataFrame.AutoVal.spatial_coherence()
```
