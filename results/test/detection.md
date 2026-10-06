# Detection results

|                   |   recall_I1_I7 |   false_positive_rate_I8 |
|:------------------|---------------:|-------------------------:|
| naive_4w_baseline |            53% |                     100% |
| stl_robust_z      |            57% |                      45% |
| forecast_pi95     |            82% |                      18% |

## Share flagged by incident type

| incident_type   |   naive_4w_baseline |   stl_robust_z |   forecast_pi95 |
|:----------------|--------------------:|---------------:|----------------:|
| I1              |                  0% |             0% |             29% |
| I2              |                 86% |            86% |            100% |
| I3              |                100% |           100% |            100% |
| I4              |                 43% |            71% |             86% |
| I5              |                 14% |            14% |             86% |
| I6              |                100% |           100% |            100% |
| I7              |                 29% |            29% |             71% |
| I8              |                100% |            45% |             18% |
