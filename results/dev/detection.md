# Detection results

|                   |   recall_I1_I7 |   false_positive_rate_I8 |
|:------------------|---------------:|-------------------------:|
| naive_4w_baseline |            57% |                     100% |
| stl_robust_z      |            55% |                      64% |
| forecast_pi95     |            92% |                       9% |

## Share flagged by incident type

| incident_type   |   naive_4w_baseline |   stl_robust_z |   forecast_pi95 |
|:----------------|--------------------:|---------------:|----------------:|
| I1              |                 43% |             0% |             71% |
| I2              |                 57% |            57% |            100% |
| I3              |                100% |           100% |            100% |
| I4              |                 14% |            29% |             71% |
| I5              |                 14% |            14% |            100% |
| I6              |                100% |           100% |            100% |
| I7              |                 71% |            86% |            100% |
| I8              |                100% |            64% |              9% |
