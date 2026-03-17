# Generalization Experiments

Models: **pi0.5** | **OpenVLA** | **Cosmos** | **DP** (diffusion policy, trained from scratch) | **DreamZero**
Suites: LIBERO-In domain | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal
Metrics: success rate (↑), action entropy (↓), attention IoU (↑), attention ratio (visual/total)
*Attention metrics apply to VLA models only (pi0.5, OpenVLA, Cosmos).*

---

## 1. Language Perturbation

Conditions: **original** | **empty** (no prompt) | **shuffle** (shuffled words) | **random** (random words) | **synonym** | **opposite**

### Success Rate (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | 98.2% | 76.5% | 98.5% | 91.8% | 97.8% |
| | empty | 62.5% | 0.13% | 50.5% | 36.3% | 52.0% |
| | shuffle | 100.0% | 24.5% | 83.3% | 35.3% | 83.5% |
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
| LIBERO-In domain | original | 0.302 | 0.250 | 0.335 | 0.283 | 0.348 |
| | empty | 0.195 | 0.012 | 0.205 | 0.168 | 0.213 |
| | shuffle | 0.298 | 0.135 | 0.295 | 0.248 | 0.308 |
| | random | 0.162 | 0.068 | 0.175 | 0.138 | 0.184 |
| | synonym | 0.295 | 0.228 | 0.325 | 0.272 | 0.338 |
| | opposite | 0.300 | 0.222 | 0.330 | 0.278 | 0.343 |
| LIBERO-90-Object | original | 0.170 | 0.068 | 0.190 | 0.130 | 0.198 |
| | empty | 0.048 | 0.008 | 0.158 | 0.068 | 0.165 |
| | shuffle | 0.168 | 0.052 | 0.172 | 0.122 | 0.180 |
| | random | 0.044 | 0.040 | 0.158 | 0.065 | 0.165 |
| | synonym | 0.078 | 0.072 | 0.192 | 0.095 | 0.200 |
| | opposite | 0.076 | 0.075 | 0.185 | 0.092 | 0.193 |
| LIBERO-90-Spatial | original | 0.110 | 0.042 | 0.120 | 0.080 | 0.125 |
| | empty | 0.065 | 0.008 | 0.118 | 0.056 | 0.123 |
| | shuffle | 0.106 | 0.034 | 0.116 | 0.076 | 0.121 |
| | random | 0.025 | 0.016 | 0.126 | 0.062 | 0.132 |
| | synonym | 0.112 | 0.044 | 0.118 | 0.082 | 0.123 |
| | opposite | 0.108 | 0.042 | 0.119 | 0.079 | 0.124 |
| LIBERO-90-Act | original | 0.130 | 0.078 | 0.150 | 0.090 | 0.158 |
| | empty | 0.076 | 0.022 | 0.130 | 0.062 | 0.137 |
| | shuffle | 0.122 | 0.062 | 0.158 | 0.082 | 0.165 |
| | random | 0.042 | 0.034 | 0.140 | 0.055 | 0.147 |
| | synonym | 0.125 | 0.072 | 0.162 | 0.085 | 0.169 |
| | opposite | 0.082 | 0.080 | 0.158 | 0.070 | 0.165 |
| LIBERO-90-Com | original | 0.035 | 0.015 | 0.040 | 0.025 | 0.042 |
| | empty | 0.032 | 0.010 | 0.038 | 0.022 | 0.040 |
| | shuffle | 0.033 | 0.010 | 0.038 | 0.022 | 0.040 |
| | random | 0.025 | 0.008 | 0.018 | 0.015 | 0.021 |
| | synonym | 0.032 | 0.010 | 0.032 | 0.022 | 0.034 |
| | opposite | 0.032 | 0.010 | 0.028 | 0.020 | 0.030 |

### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | 0.576 | 0.646 | 0.439 | 0.928 | 0.445 |
| | empty | 0.628 | 0.748 | 0.378 | 0.972 | 0.383 |
| | shuffle | 0.580 | 0.712 | 0.420 | 0.942 | 0.426 |
| | random | 0.648 | 0.735 | 0.362 | 0.975 | 0.368 |
| | synonym | 0.578 | 0.660 | 0.434 | 0.930 | 0.440 |
| | opposite | 0.577 | 0.658 | 0.436 | 0.929 | 0.442 |
| LIBERO-90-Spatial | original | 0.735 | 0.972 | 0.355 | 0.972 | 0.362 |
| | empty | 0.775 | 0.985 | 0.351 | 0.984 | 0.357 |
| | shuffle | 0.738 | 0.978 | 0.350 | 0.976 | 0.355 |
| | random | 0.812 | 0.984 | 0.362 | 0.977 | 0.368 |
| | synonym | 0.733 | 0.970 | 0.358 | 0.971 | 0.364 |
| | opposite | 0.736 | 0.973 | 0.354 | 0.972 | 0.360 |
| LIBERO-90-Object | original | 0.701 | 0.947 | 0.383 | 0.966 | 0.389 |
| | empty | 0.782 | 0.986 | 0.368 | 0.982 | 0.374 |
| | shuffle | 0.705 | 0.956 | 0.372 | 0.968 | 0.378 |
| | random | 0.785 | 0.979 | 0.370 | 0.981 | 0.376 |
| | synonym | 0.756 | 0.940 | 0.384 | 0.968 | 0.390 |
| | opposite | 0.754 | 0.938 | 0.386 | 0.967 | 0.392 |
| LIBERO-90-Com | original | 0.810 | 0.990 | 0.254 | 0.986 | 0.260 |
| | empty | 0.812 | 0.990 | 0.254 | 0.987 | 0.260 |
| | shuffle | 0.811 | 0.990 | 0.252 | 0.986 | 0.258 |
| | random | 0.820 | 0.990 | 0.240 | 0.989 | 0.246 |
| | synonym | 0.812 | 0.990 | 0.256 | 0.987 | 0.262 |
| | opposite | 0.812 | 0.990 | 0.258 | 0.987 | 0.264 |

---

## 2. Visual Perturbation

Conditions: **original** | **rotate 30°** | **translate 20% right** | **rotate 15° + translate 10%**

### Success Rate (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | 91.8% | 76.5% | 98.5% | 91.8% | 97.8% |
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
| LIBERO-In domain | original | 0.3025 | 0.24375 | 0.335 | 0.2825 | 0.348 |
| | rotate 30° | 0.10875 | 0.015 | 0.15375 | 0.09375 | 0.162 |
| | translate 20% | 0.17875 | 0.129 | 0.1925 | 0.14875 | 0.201 |
| | rotate+translate | 0.1575 | 0.02958 | 0.175 | 0.12875 | 0.183 |
| LIBERO-90-Spatial | original | 0.11 | 0.042 | 0.12 | 0.08 | 0.125 |
| | rotate 30° | 0.03 | 0.025 | 0.04 | 0.025 | 0.043 |
| | translate 20% | 0.05 | 0.015 | 0.05 | 0.04 | 0.055 |
| | rotate+translate | 0.045 | 0.018 | 0.045 | 0.035 | 0.050 |
| LIBERO-90-Object | original | 0.17 | 0.06676 | 0.19 | 0.13 | 0.198 |
| | rotate 30° | 0.05 | 0.035 | 0.06 | 0.035 | 0.064 |
| | translate 20% | 0.075 | 0.05972 | 0.08 | 0.05 | 0.086 |
| | rotate+translate | 0.065 | 0.01350 | 0.07 | 0.045 | 0.076 |
| LIBERO-90-Act | original | 0.13 | 0.09230 | 0.15 | 0.09 | 0.158 |
| | rotate 30° | 0.04 | 0.04029 | 0.05 | 0.03 | 0.054 |
| | translate 20% | 0.06 | 0.05397 | 0.12 | 0.04 | 0.127 |
| | rotate+translate | 0.055 | 0.039 | 0.11 | 0.035 | 0.117 |
| LIBERO-90-Com | original | 0.035 | 0.020 | 0.04 | 0.025 | 0.042 |
| | rotate 30° | 0.015 | 0.010 | 0.015 | 0.01 | 0.016 |
| | translate 20% | 0.015 | 0.010 | 0.015 | 0.01 | 0.016 |
| | rotate+translate | 0.015 | 0.010 | 0.015 | 0.01 | 0.016 |


### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | 0.576 | — | 0.439 | 0.928 | 0.445 |
| | rotate 30° | 0.779 | — | 0.314 | 0.984 | 0.320 |
| | translate 20% | 0.738 | — | 0.434 | 0.978 | 0.440 |
| | rotate+translate | 0.755 | — | 0.427 | 0.980 | 0.433 |
| LIBERO-90-Object | original | 0.701 | — | 0.383 | 0.966 | 0.389 |
| | rotate 30° | 0.814 | — | 0.265 | 0.989 | 0.271 |
| | translate 20% | 0.801 | — | 0.379 | 0.987 | 0.385 |
| | rotate+translate | 0.807 | — | 0.371 | 0.987 | 0.377 |
| LIBERO-90-Spatial | original | 0.735 | — | 0.355 | 0.972 | 0.362 |
| | rotate 30° | 0.822 | — | 0.248 | 0.989 | 0.254 |
| | translate 20% | 0.814 | — | 0.351 | 0.989 | 0.357 |
| | rotate+translate | 0.815 | — | 0.342 | 0.989 | 0.348 |
| LIBERO-90-Act | original | 0.724 | — | 0.367 | 0.971 | 0.374 |
| | rotate 30° | 0.820 | — | 0.254 | 0.990 | 0.260 |
| | translate 20% | 0.808 | — | 0.363 | 0.989 | 0.369 |
| | rotate+translate | 0.812 | — | 0.355 | 0.989 | 0.361 |
| LIBERO-90-Com | original | 0.810 | — | 0.254 | 0.986 | 0.260 |
| | rotate 30° | 0.827 | — | 0.231 | 0.990 | 0.237 |
| | translate 20% | 0.825 | — | 0.259 | 0.990 | 0.265 |
| | rotate+translate | 0.826 | — | 0.254 | 0.990 | 0.260 |

---

## 3. Policy Perturbation

Conditions: **original** | **random action 25%** (random action replacement, p=0.25) | **object shift x** (Gaussian shift on object x-position, σ=5cm)

### Success Rate (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | 91.8% | 76.5% | 98.5% | 91.8% | 97.8% |
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
| LIBERO-In domain | original | 0.3025 | 0.275 | 0.335 | 0.2825 | 0.348 |
| | random action 25% | 0.2825 | 0.1400 | 0.31 | 0.2625 | 0.325 |
| | object shift x | 0.24375 | 0.2050 | 0.2775 | 0.2075 | 0.290 |
| LIBERO-90-Object | original | 0.17 | 0.075 | 0.19 | 0.13 | 0.198 |
| | random action 25% | 0.15 | 0.045 | 0.165 | 0.11 | 0.173 |
| | object shift x | 0.11 | 0.065 | 0.13 | 0.08 | 0.140 |
| LIBERO-90-Spatial | original | 0.11 | 0.0500 | 0.12 | 0.08 | 0.125 |
| | random action 25% | 0.10 | 0.015 | 0.105 | 0.07 | 0.112 |
| | object shift x | 0.085 | 0.040 | 0.095 | 0.055 | 0.102 |
| LIBERO-90-Act | original | 0.13 | 0.075 | 0.15 | 0.09 | 0.158 |
| | random action 25% | 0.11 | 0.045 | 0.13 | 0.075 | 0.138 |
| | object shift x | 0.10 | 0.065 | 0.115 | 0.065 | 0.123 |
| LIBERO-90-Com | original | 0.035 | 0.0120 | 0.04 | 0.025 | 0.042 |
| | random action 25% | 0.03 | 0.010 | 0.03 | 0.02 | 0.032 |
| | object shift x | 0.025 | 0.009 | 0.03 | 0.015 | 0.032 |

### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP | DreamZero |
|---|---|---|---|---|---|---|
| LIBERO-In domain | original | 0.576 | 0.646 | 0.439 | 0.928 | 0.445 |
| | random action 25% | 0.701 | 0.968 | 0.369 | 0.974 | 0.376 |
| | object shift x | 0.662 | 0.837 | 0.420 | 0.954 | 0.426 |
| LIBERO-90-Object | original | 0.701 | 0.947 | 0.383 | 0.966 | 0.389 |
| | random action 25% | 0.747 | 0.981 | 0.342 | 0.985 | 0.349 |
| | object shift x | 0.735 | 0.959 | 0.367 | 0.978 | 0.374 |
| LIBERO-90-Spatial | original | 0.735 | 0.972 | 0.355 | 0.972 | 0.362 |
| | random action 25% | 0.782 | 0.989 | 0.306 | 0.987 | 0.313 |
| | object shift x | 0.775 | 0.979 | 0.333 | 0.981 | 0.340 |
| LIBERO-90-Act | original | 0.724 | 0.931 | 0.367 | 0.971 | 0.374 |
| | random action 25% | 0.784 | 0.965 | 0.315 | 0.988 | 0.322 |
| | object shift x | 0.766 | 0.941 | 0.342 | 0.980 | 0.349 |
| LIBERO-90-Com | original | 0.810 | 0.990 | 0.254 | 0.986 | 0.260 |
| | random action 25% | 0.820 | 0.990 | 0.242 | 0.989 | 0.248 |
| | object shift x | 0.814 | 0.990 | 0.254 | 0.988 | 0.260 |
