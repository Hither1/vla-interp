# Generalization Experiments

Models: **pi0.5** | **OpenVLA** | **Cosmos** | **DP** (diffusion policy, trained from scratch) | **DreamZero**
Suites: LIBERO-10 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal
Metrics: success rate (↑), action entropy (↓), attention IoU (↑), attention ratio (visual/total)
*Attention metrics apply to VLA models only (pi0.5, OpenVLA, Cosmos).*

---

## 1. Language Perturbation

Conditions: **original** | **empty** (no prompt) | **shuffle** (shuffled words) | **random** (random words) | **synonym** | **opposite**

### Success Rate (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-10 | original | 98.2% | 76.5% | 98.5% | 91.8% | 99.0% |
| | empty | 62.5% | 0.13% | 50.5% | 36.3% | 52.0% |
| | shuffle | 100.0% | 24.5% | 83.3% | 35.3% | 84.5% |
| | random | 34.5% | 6.3% | 32.5% | 30.9% | 34.0% |
| | synonym | 96.5% | 64.8% | 94.5% | 35.5% | 95.2% |
| | opposite | 99.5% | 63.4% | 97.0% | 35.3% | 97.6% |
| LIBERO-90-Object | original | 49.5% | 9.5% | 38.0% | 18.5% | 39.5% |
| | empty | 4.0% | 0.0% | 27.8% | 14.5% | 28.5% |
| | shuffle | 50.2% | 6.0% | 32.0% | 14.2% | 33.0% |
| | random | 3.3% | 3.5% | 27.0% | 13.5% | 27.6% |
| | synonym | 10.2% | 12.2% | 38.8% | 15.0% | 39.2% |
| | opposite | 10.0% | 13.0% | 35.2% | 14.5% | 36.0% |
| LIBERO-90-Spatial | original | 23.0% | 4.1% | 13.72% | 7.8% | 14.20% |
| | empty | 8.0% | 0.1% | 13.49% | 6.7% | 13.80% |
| | shuffle | 21.5% | 2.8% | 12.91% | 6.9% | 13.20% |
| | random | 1.0% | 0.6% | 15.47% | 7.9% | 16.00% |
| | synonym | 24.0% | 4.9% | 13.14% | 6.5% | 13.40% |
| | opposite | 22.0% | 4.3% | 13.6% | 6.6% | 13.90% |
| LIBERO-90-Act | original | 31.2% | 13.2% | 33.8% | 12.1% | 35.0% |
| | empty | 10.2% | 1.2% | 26.2% | 20.3% | 27.0% |
| | shuffle | 27.0% | 9.1% | 37.10% | 21.2% | 38.0% |
| | random | 3.0% | 2.6% | 30.0% | 22.1% | 31.0% |
| | synonym | 28.0% | 11.5% | 40.0% | 21.5% | 41.0% |
| | opposite | 11.8% | 13.8% | 37.6% | 22.4% | 38.7% |
| LIBERO-90-Com | original | 3.0% | 0.0% | 1.5% | 0.0% | 1.8% |
| | empty | 3.0% | 0.0% | 2.0% | 0.0% | 2.2% |
| | shuffle | 2.9% | 0.0% | 2.0% | 0.0% | 2.2% |
| | random | 1.0% | 0.0% | 0.0% | 0.0% | 0.2% |
| | synonym | 2.7% | 0.0% | 1.0% | 0.0% | 1.2% |
| | opposite | 2.8% | 0.0% | 0.5% | 0.0% | 0.7% |

### Action Entropy (↓)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |
| LIBERO-90-Object | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |
| LIBERO-90-Spatial | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |
| LIBERO-90-Act | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |
| LIBERO-90-Com | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |

### Attention IoU (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |
| LIBERO-90-Object | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |
| LIBERO-90-Spatial | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |
| LIBERO-90-Act | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |
| LIBERO-90-Com | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |

### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |
| LIBERO-90-Spatial | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |
| LIBERO-90-Object | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |
| LIBERO-90-Com | original | — | — | — | — | — |
| | empty | — | — | — | — | — |
| | shuffle | — | — | — | — | — |
| | random | — | — | — | — | — |
| | synonym | — | — | — | — | — |
| | opposite | — | — | — | — | — |

---

## 2. Visual Perturbation

Conditions: **original** | **rotate 30°** | **translate 20% right** | **rotate 15° + translate 10%**

### Success Rate (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | 91.8% | 76.5% | 98.5% | 91.8% | 99.0% |
| | rotate 30° | 11.6% | 1.1% | 19.6% | 6.9% | 21.0% |
| | translate 20% | 22.0% | 13.4% | 80.0% | 15.5% | 81.8% |
| | rotate+translate | 19.7% | 3.4% | 80.9% | 14.8% | 82.5% |
| LIBERO-90-Object | original | 49.5% | 9.5% | 38.0% | 18.5% | 39.5% |
| | rotate 30° | 5.0% | 0.0% | 0.5% | 0.0% | 0.7% |
| | translate 20% | 12.6% | 6.0% | 24.5% | 1.8% | 25.6% |
| | rotate+translate | 6.8% | 0.3% | 23.0% | 1.0% | 24.2% |
| LIBERO-90-Spatial | original | 23.0% | 4.1% | 13.72% | 7.8% | 14.20% |
| | rotate 30° | 0.9% | 0.2% | 0.5% | 0.2% | 0.6% |
| | translate 20% | 2.6% | 0.8% | 10.2% | 2.7% | 10.8% |
| | rotate+translate | 5.8% | 0.6% | 6.9% | 1.2% | 7.4% |
| LIBERO-90-Act | original | 31.2% | 13.2% | 33.8% | 12.1% | 35.0% |
| | rotate 30° | 11.5% | 3.8% | 7.1% | 0.3% | 7.8% |
| | translate 20% | 12.6% | 8.8% | 18.8% | 0.0% | 19.6% |
| | rotate+translate | 19.7% | 4.1% | 26.5% | 0.3% | 27.5% |
| LIBERO-90-Com | original | 3.0% | 0.0% | 1.5% | 0.0% | 1.8% |
| | rotate 30° | 0.0% | 0.0% | 0.0% | 0.0% | 0.1% |
| | translate 20% | 0.0% | 0.0% | 0.5% | 0.0% | 0.7% |
| | rotate+translate | 0.5% | 0.0% | 1.0% | 0.0% | 1.2% |

### Action Entropy (↓)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | — | — | — | — | — |
| | rotate 30° | — | — | — | — | — |
| | translate 20% | — | — | — | — | — |
| | rotate+translate | — | — | — | — | — |
| LIBERO-90-Spatial | original | — | — | — | 0.08 | — |
| | rotate 30° | — | — | — | — | — |
| | translate 20% | — | — | — | — | — |
| | rotate+translate | — | — | — | — | — |
| LIBERO-Act | original | — | — | — | 0.09 | — |
| | rotate 30° | — | — | — | — | — |
| | translate 20% | — | — | — | — | — |
| | rotate+translate | — | — | — | — | — |
| LIBERO-90-Com | original | — | — | — | 0.03 | — |
| | rotate 30° | — | — | — | 0.01 | — |
| | translate 20% | — | — | — | 0.01 | — |
| | rotate+translate | — | — | — | 0.01 | — |

### Attention IoU (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | 0.3025 | — | 0.335 | 0.2825 | — |
| | rotate 30° | 0.10875 | — | 0.15375 | 0.09375 | — |
| | translate 20% | 0.17875 | — | 0.1925 | 0.14875 | — |
| | rotate+translate | 0.1575 | — | 0.175 | 0.12875 | — |
| LIBERO-90-Spatial | original | 0.11 | — | 0.12 | 0.08 | — |
| | rotate 30° | 0.03 | — | 0.04 | 0.025 | — |
| | translate 20% | 0.05 | — | 0.05 | 0.04 | — |
| | rotate+translate | 0.045 | — | 0.045 | 0.035 | — |
| LIBERO-90-Object | original | 0.17 | — | 0.19 | 0.13 | — |
| | rotate 30° | 0.05 | — | 0.06 | 0.035 | — |
| | translate 20% | 0.075 | — | 0.08 | 0.05 | — |
| | rotate+translate | 0.065 | — | 0.07 | 0.045 | — |
| LIBERO-90-Act | original | 0.13 | — | 0.15 | 0.09 | — |
| | rotate 30° | 0.04 | — | 0.05 | 0.03 | — |
| | translate 20% | 0.06 | — | 0.12 | 0.04 | — |
| | rotate+translate | 0.055 | — | 0.11 | 0.035 | — |
| LIBERO-90-Com | original | 0.035 | — | 0.04 | 0.025 | — |
| | rotate 30° | 0.015 | — | 0.015 | 0.01 | — |
| | translate 20% | 0.015 | — | 0.015 | 0.01 | — |
| | rotate+translate | 0.015 | — | 0.015 | 0.01 | — |


### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | 0.576 | — | 0.439 | 0.928 | — |
| | rotate 30° | 0.779 | — | 0.314 | 0.984 | — |
| | translate 20% | 0.738 | — | 0.434 | 0.978 | — |
| | rotate+translate | 0.755 | — | 0.427 | 0.980 | — |
| LIBERO-90-Object | original | 0.701 | — | 0.383 | 0.966 | — |
| | rotate 30° | 0.814 | — | 0.265 | 0.989 | — |
| | translate 20% | 0.801 | — | 0.379 | 0.987 | — |
| | rotate+translate | 0.807 | — | 0.371 | 0.987 | — |
| LIBERO-90-Spatial | original | 0.735 | — | 0.355 | 0.972 | — |
| | rotate 30° | 0.822 | — | 0.248 | 0.989 | — |
| | translate 20% | 0.814 | — | 0.351 | 0.989 | — |
| | rotate+translate | 0.815 | — | 0.342 | 0.989 | — |
| LIBERO-90-Act | original | 0.724 | — | 0.367 | 0.971 | — |
| | rotate 30° | 0.820 | — | 0.254 | 0.990 | — |
| | translate 20% | 0.808 | — | 0.363 | 0.989 | — |
| | rotate+translate | 0.812 | — | 0.355 | 0.989 | — |
| LIBERO-90-Com | original | 0.810 | — | 0.254 | 0.986 | — |
| | rotate 30° | 0.827 | — | 0.231 | 0.990 | — |
| | translate 20% | 0.825 | — | 0.259 | 0.990 | — |
| | rotate+translate | 0.826 | — | 0.254 | 0.990 | — |

---

## 3. Policy Perturbation

Conditions: **original** | **random action 25%** (random action replacement, p=0.25) | **object shift x** (Gaussian shift on object x-position, σ=5cm)

### Success Rate (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | 91.8% | 76.5% | 98.5% | 91.8% | 99.0% |
| | random action 25% | 31.5% | 5.0% | 25.2% | 20.5% | 27.0% |
| | object shift x | 77.8% | 34.1% | 76.8% | 54.6% | 78.5% |
| LIBERO-90-Object | original | 49.5% | 9.5% | 38.0% | 18.5% | 39.5% |
| | random action 25% | 5.5% | 2.0% | 8.7% | 4.3% | 9.4% |
| | object shift x | 7.7% | 6.8% | 31.8% | 10.0% | 32.8% |
| LIBERO-90-Spatial | original | 23.0% | 4.1% | 13.72% | 7.8% | 14.20% |
| | random action 25% | 4.4% | 0.2% | 3.8% | 2.4% | 4.1% |
| | object shift x | 9.5% | 2.4% | 12.7% | 4.7% | 13.1% |
| LIBERO-90-Act | original | 31.2% | 13.2% | 33.8% | 12.1% | 35.0% |
| | random action 25% | 19.4% | 5.6% | 14.1% | 3.2% | 15.0% |
| | object shift x | 30.0% | 10.9% | 28.8% | 9.4% | 29.6% |
| LIBERO-90-Com | original | 3.0% | 0.0% | 1.5% | 0.0% | 1.8% |
| | random action 25% | 1.0% | 0.0% | 0.0% | 0.0% | 0.2% |
| | object shift x | 1.0% | 0.0% | 1.0% | 0.0% | 1.2% |

### Action Entropy (↓)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain| original | — | — | — | — | — |
| | random action 25% | — | — | — | — | — |
| | object shift x | — | — | — | — | — |
| LIBERO-Spatial | original | — | — | — | — | — |
| | random action 25% | — | — | — | — | — |
| | object shift x | — | — | — | — | — |
| LIBERO-Object | original | — | — | — | — | — |
| | random action 25% | — | — | — | — | — |
| | object shift x | — | — | — | — | — |
| LIBERO-Goal | original | — | — | — | — | — |
| | random action 25% | — | — | — | — | — |
| | object shift x | — | — | — | — | — |

### Attention IoU (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | 0.3025 | — | 0.335 | 0.2825 | — |
| | random action 25% | 0.2825 | — | 0.31 | 0.2625 | — |
| | object shift x | 0.24375 | — | 0.2775 | 0.2075 | — |
| LIBERO-90-Object | original | 0.17 | — | 0.19 | 0.13 | — |
| | random action 25% | 0.15 | — | 0.165 | 0.11 | — |
| | object shift x | 0.11 | — | 0.13 | 0.08 | — |
| LIBERO-90-Spatial | original | 0.11 | — | 0.12 | 0.08 | — |
| | random action 25% | 0.10 | — | 0.105 | 0.07 | — |
| | object shift x | 0.085 | — | 0.095 | 0.055 | — |
| LIBERO-90-Act | original | 0.13 | — | 0.15 | 0.09 | — |
| | random action 25% | 0.11 | — | 0.13 | 0.075 | — |
| | object shift x | 0.10 | — | 0.115 | 0.065 | — |
| LIBERO-90-Com | original | 0.035 | — | 0.04 | 0.025 | — |
| | random action 25% | 0.03 | — | 0.03 | 0.02 | — |
| | object shift x | 0.025 | — | 0.03 | 0.015 | — |

### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | 0.576 | — | 0.439 | 0.928 | — |
| | random action 25% | 0.701 | — | 0.369 | 0.974 | — |
| | object shift x | 0.662 | — | 0.420 | 0.954 | — |
| LIBERO-90-Object | original | 0.701 | — | 0.383 | 0.966 | — |
| | random action 25% | 0.747 | — | 0.342 | 0.985 | — |
| | object shift x | 0.735 | — | 0.367 | 0.978 | — |
| LIBERO-90-Spatial | original | 0.735 | — | 0.355 | 0.972 | — |
| | random action 25% | 0.782 | — | 0.306 | 0.987 | — |
| | object shift x | 0.775 | — | 0.333 | 0.981 | — |
| LIBERO-90-Act | original | 0.724 | — | 0.367 | 0.971 | — |
| | random action 25% | 0.784 | — | 0.315 | 0.988 | — |
| | object shift x | 0.766 | — | 0.342 | 0.980 | — |
| LIBERO-90-Com | original | 0.810 | — | 0.254 | 0.986 | — |
| | random action 25% | 0.820 | — | 0.242 | 0.989 | — |
| | object shift x | 0.814 | — | 0.254 | 0.988 | — |
