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

All the tests add a column to the original pd.DataFrame. This columns indicatees data that does not pass the test when it is 1, and data that does it whe it is 0. 


**Impossible values**

Label values too big or too small to be physically possible. The labels are added in a column with the suffix '_IV'.

```python
import autoval.utils
import autoval.validation_tests

file_path = './data/station_name/'
variables_to_validate = ['TMPA', 'WSDP', 'RADS01']

# Open all data from a station
observations = autoval.utils.open_observations(file_path, variables_to_validate)

# Validate
observations = observations.AutoVal.impossible_values(variables_to_validate)`
```

**Climatological coherence**

Label values that are outside the selected climatological percentile gap. The climatology is calculated for every month and every hour. This means that the selected percentile thresholds of the daily cycle of the variable is calculated for each month. If a value is outside the gap created by the upper percentile daily cycle and the lower percentile daily cycle, it is labeled in a column with the suffix '_CC'.

```python
percentile_thresholds = [0.01, 0.99]

# Validate
observations = observations.AutoVal.climatological_coherence(variables_to_validate, percentile_thresholds)`
```


**Temporal coherence**

Work similarly to the climatological coherence test, but instead of the values, it uses the difference with the previous value. It labels values that change too much or too little compared with what is climatologically expected. The labels are added in a column with the suffix '_TC'.

```python
percentile_thresholds = [0.01, 0.99]

# Validate
observations = observations.AutoVal.temporal_coherence(variables_to_validate, percentile_thresholds)`
```

**Spatial coherence**

It calculates the correlation and linear regression for each hour of each month between two nearby meteorological stations. The predicted value of the linear regression using the reference station as a predictor is saved. When the correlation is above a certain threshold, it saves the residuals of the regression. When a residual is outside the percentile gap (The observation is too far away from the expected regression) the value is labeled in a column with the suffix '_SC'.

```python
percentile_thresholds = [0.01, 0.99]

file_path_reference = './data/reference_station_name/'

# Open data from a reference station
observations = autoval.utils.open_observations(file_path_reference, variables_to_validate)

# Validate
observations = observations.AutoVal.spatial_coherence(reference_observations, variables_to_validate, percentile_thresholds)`
```


