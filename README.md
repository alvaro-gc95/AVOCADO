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
In order to be read by AutoVal, the required format of the original files is the following:
```
Put here image of the file when possible
```
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
