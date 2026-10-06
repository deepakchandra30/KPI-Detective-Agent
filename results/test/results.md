# Benchmark results

| system | scenarios | top1_accuracy | top3_accuracy | top1_segment_only | classification_accuracy | false_alarm_handling | unsupported_claim_rate | avg_steps | avg_cost_usd | avg_latency_s | consistency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rule_based | 60 | 0.0% | 0.0% | 41.7% | 58.3% | 0.0% | 0.0% | 5.0 | $0.0000 | 0.08 | n/a |
| playbook | 60 | 85.0% | 76.7% | 86.7% | 86.7% | 72.7% | 0.0% | 7.9 | $0.0000 | 0.27 | n/a |
| playbook_no_critic | 60 | 85.0% | 76.7% | 86.7% | 86.7% | 72.7% | 0.0% | 7.9 | $0.0000 | 0.26 | n/a |

## Top-1 accuracy by incident type

| incident_type   |   rule_based |   rule_based (segment only) |   playbook |   playbook_no_critic |
|:----------------|-------------:|----------------------------:|-----------:|---------------------:|
| I1              |           0% |                          0% |        86% |                  86% |
| I2              |           0% |                        100% |       100% |                 100% |
| I3              |           0% |                        100% |       100% |                 100% |
| I4              |           0% |                        100% |       100% |                 100% |
| I5              |           0% |                          0% |        57% |                  57% |
| I6              |           0% |                          0% |       100% |                 100% |
| I7              |           0% |                         57% |        71% |                  71% |
| I8              |           0% |                          0% |        73% |                  73% |

## Top-1 accuracy by incident magnitude (I8 excluded)

| difficulty   |   rule_based |   playbook |   playbook_no_critic |
|:-------------|-------------:|-----------:|---------------------:|
| small        |           0% |        72% |                  72% |
| medium       |           0% |        88% |                  88% |
| large        |           0% |       100% |                 100% |
