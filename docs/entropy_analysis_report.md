# Action Entropy Analysis by LIBERO Suite

This report analyzes action entropy across different LIBERO task suites, comparing successful and failed trial runs.

## Understanding Entropy

**Action Entropy** measures the randomness/unpredictability of robot actions:

- **Calculated using**: Kernel Density Estimation (KDE) on action sequences
- **Values**: Negative numbers (entropy = -mean(log_density))
- **Interpretation**:
  - More negative (e.g., -10.0) = **Lower entropy** = More deterministic/consistent actions
  - Less negative (e.g., -6.0) = **Higher entropy** = More random/exploratory actions

---

## Results by Suite

### LIBERO-10

**Tasks**: 19  
**Success trials**: 368  
**Failure trials**: 10  
**Success rate**: 97.4%

#### Success Trials

| Metric | Value |
|--------|-------|
| Mean entropy | -9.0294 |
| Std deviation | 1.8696 |
| Min entropy | -15.6325 |
| Max entropy | -6.1739 |

#### Failure Trials

| Metric | Value |
|--------|-------|
| Mean entropy | -8.5972 |
| Std deviation | 1.2314 |
| Min entropy | -10.8560 |
| Max entropy | -6.8599 |

#### Comparison

**Entropy difference** (Failure - Success): `0.4321`

**Failed trials have 0.4321 HIGHER entropy** (more random):
- Policy shows more exploratory/uncertain behavior during failures
- May indicate the policy is "searching" for solutions when struggling

#### Task Details

| Task | Success | Failure | S. Entropy | F. Entropy | Diff |
|------|---------|---------|------------|------------|------|
| pick up the black bowl in the top drawer... | 20 | 0 | -6.973 | N/A | N/A |
| pick up the orange juice and place it in... | 19 | 1 | -9.904 | -10.324 | -0.420 |
| open the middle drawer of the cabinet | 20 | 0 | -12.416 | N/A | N/A |
| pick up the black bowl between the plate... | 20 | 0 | -8.178 | N/A | N/A |
| pick up the salad dressing and place it ... | 20 | 0 | -9.983 | N/A | N/A |
| pick up the black bowl next to the cooki... | 20 | 0 | -8.616 | N/A | N/A |
| pick up the bbq sauce and place it in th... | 20 | 0 | -9.583 | N/A | N/A |
| open the top drawer and put the bowl ins... | 20 | 0 | -7.255 | N/A | N/A |
| push the plate to the front of the stove | 20 | 0 | -14.325 | N/A | N/A |
| pick up the black bowl next to the plate... | 19 | 1 | -7.602 | -6.860 | +0.742 |

*Showing top 10 of 19 tasks*

---

### LIBERO-90

**Tasks**: 14  
**Success trials**: 52  
**Failure trials**: 228  
**Success rate**: 18.6%

#### Success Trials

| Metric | Value |
|--------|-------|
| Mean entropy | -8.9964 |
| Std deviation | 0.9529 |
| Min entropy | -10.7942 |
| Max entropy | -7.3010 |

#### Failure Trials

| Metric | Value |
|--------|-------|
| Mean entropy | -9.6612 |
| Std deviation | 0.9792 |
| Min entropy | -13.1669 |
| Max entropy | -7.3123 |

#### Comparison

**Entropy difference** (Failure - Success): `-0.6648`

**Failed trials have 0.6648 LOWER entropy** (more deterministic):
- Policy is confident but systematically wrong
- Suggests learned incorrect behavior patterns

#### Task Details

| Task | Success | Failure | S. Entropy | F. Entropy | Diff |
|------|---------|---------|------------|------------|------|
| put the yellow and white mug on the righ... | 20 | 0 | -8.613 | N/A | N/A |
| put the red mug on the right plate | 0 | 20 | N/A | -10.093 | N/A |
| put the yellow and white mug to the fron... | 0 | 20 | N/A | -8.649 | N/A |
| put the middle black bowl on top of the ... | 0 | 20 | N/A | -9.399 | N/A |
| put the frying pan on top of the cabinet | 0 | 20 | N/A | -10.099 | N/A |
| put the white bowl to the right of the p... | 0 | 20 | N/A | -10.167 | N/A |
| put the white mug on the left plate | 18 | 2 | -10.008 | -10.275 | -0.267 |
| put the frying pan on the cabinet shelf | 0 | 20 | N/A | -9.348 | N/A |
| put the red mug on the left plate | 0 | 20 | N/A | -10.021 | N/A |
| put the right moka pot on the stove | 13 | 7 | -8.070 | -8.481 | -0.411 |

*Showing top 10 of 14 tasks*

---

## Overall Summary

### Suite Comparison Table

| Suite | Success Rate | Success Entropy | Failure Entropy | Difference | Interpretation |
|-------|--------------|-----------------|-----------------|------------|----------------|
| LIBERO-10 | 97.4% | -9.029 | -8.597 | +0.432 | More random in failure |
| LIBERO-90 | 18.6% | -8.996 | -9.661 | -0.665 | More deterministic in failure |

### Key Insights

1. **Average success entropy**: -9.0129
2. **Average failure entropy**: -9.1292
3. **Average difference**: -0.1164

**Overall pattern**: Failed trials tend to have more deterministic actions. This suggests the policy is confident but systematically wrong when it fails.

## Recommendations

Based on the entropy analysis:

- **LIBERO-90**: Low entropy in failures suggests the policy has learned incorrect patterns - consider analyzing failure modes and retraining

---

*Report generated using KDE-based action entropy analysis*
