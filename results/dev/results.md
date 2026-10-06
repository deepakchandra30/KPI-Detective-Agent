# Benchmark results

| system | scenarios | top1_accuracy | top3_accuracy | top1_segment_only | classification_accuracy | false_alarm_handling | unsupported_claim_rate | avg_steps | avg_cost_usd | avg_latency_s | consistency |
|---|---|---|---|---|---|---|---|---|---|---|---|
| rule_based | 60 | 0.0% | 0.0% | 38.3% | 58.3% | 0.0% | 0.0% | 5.0 | $0.0000 | 0.08 | n/a |
| playbook | 60 | 91.7% | 86.7% | 91.7% | 95.0% | 81.8% | 0.0% | 7.8 | $0.0000 | 0.27 | n/a |
| playbook_no_critic | 60 | 91.7% | 86.7% | 91.7% | 95.0% | 81.8% | 0.0% | 7.8 | $0.0000 | 0.26 | n/a |

## Top-1 accuracy by incident type

| incident_type   |   rule_based |   rule_based (segment only) |   playbook |   playbook_no_critic |
|:----------------|-------------:|----------------------------:|-----------:|---------------------:|
| I1              |           0% |                          0% |        71% |                  71% |
| I2              |           0% |                         71% |        86% |                  86% |
| I3              |           0% |                         71% |       100% |                 100% |
| I4              |           0% |                        100% |       100% |                 100% |
| I5              |           0% |                          0% |       100% |                 100% |
| I6              |           0% |                          0% |       100% |                 100% |
| I7              |           0% |                         86% |       100% |                 100% |
| I8              |           0% |                          0% |        82% |                  82% |

## Top-1 accuracy by incident magnitude (I8 excluded)

| difficulty   |   rule_based |   playbook |   playbook_no_critic |
|:-------------|-------------:|-----------:|---------------------:|
| small        |           0% |        94% |                  94% |
| medium       |           0% |        87% |                  87% |
| large        |           0% |       100% |                 100% |
