# Generalization Experiments

Models: **pi0.5** | **OpenVLA** | **Cosmos** | **DP** (diffusion policy, trained from scratch)
Suites: LIBERO-10 | LIBERO-Spatial | LIBERO-Object | LIBERO-Goal
Metrics: success rate (↑), action entropy (↓), attention IoU (↑), attention ratio (visual/total)
*Attention metrics apply to VLA models only (pi0.5, OpenVLA, Cosmos).*

---

## 1. Language Perturbation

Conditions: **original** | **empty** (no prompt) | **shuffle** (shuffled words) | **random** (random words) | **synonym** | **opposite**

### Success Rate (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP |
|---|---|---|---|---|---|
| LIBERO-10 | original | 98.2% | — | — | — |
| | empty | 62.5% | — | — | — |
| | shuffle | 100.0% | — | — | — |
| | random | 34.5% | — | — | — |
| | synonym | 96.5% | — | — | — |
| | opposite | 99.5% | — | — | — |
| LIBERO-Spatial | original | 92.4% | — | — | — |
| | empty | 75.5% | — | — | — |
| | shuffle | 93.0% | — | — | — |
| | random | 43.0% | — | — | — |
| | synonym | 96.5% | — | — | — |
| | opposite | 92.0% | — | — | — |
| LIBERO-Object | original | 98.8% | — | — | — |
| | empty | 64.0% | — | — | — |
| | shuffle | 93.0% | — | — | — |
| | random | 63.5% | — | — | — |
| | synonym | 99.0% | — | — | — |
| | opposite | 99.5% | — | — | — |
| LIBERO-Goal | original | 98.0% | — | — | — |
| | empty | 9.0% | — | — | — |
| | shuffle | 89.5% | — | — | — |
| | random | 62.0% | — | — | — |
| | synonym | 96.0% | — | — | — |
| | opposite | 97.5% | — | — | — |
| LIBERO-90-Object | original | 49.5% | — | — | — |
| | empty | 4.0% | — | — | — |
| | shuffle | 50.2% | — | — | — |
| | random | 3.3% | — | — | — |
| | synonym | 10.2% | — | — | — |
| | opposite | 10.0% | — | — | — |
| LIBERO-90-Spatial | original | 23.0% | — | — | — |
| | empty | 8.0% | — | — | — |
| | shuffle | 21.5% | — | — | — |
| | random | 1.0% | — | — | — |
| | synonym | 24.0% | — | — | — |
| | opposite | 22.0% | — | — | — |
| LIBERO-90-Act | original | 31.2% | — | — | — |
| | empty | 10.2% | — | — | — |
| | shuffle | 27.0% | — | — | — |
| | random | 3.0% | — | — | — |
| | synonym | 28.0% | — | — | — |
| | opposite | 11.8% | — | — | — |
| LIBERO-90-Com | original | 3.0% | — | — | — |
| | empty | 3.0% | — | — | — |
| | shuffle | 2.9% | — | — | — |
| | random | 1.0% | — | — | — |
| | synonym | 2.7% | — | — | — |
| | opposite | 2.8% | — | — | — |

### Action Entropy (↓)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP |
|---|---|---|---|---|---|
| LIBERO-10 | original | — | — | — | — |
| | empty | — | — | — | — |
| | shuffle | — | — | — | — |
| | random | — | — | — | — |
| | synonym | — | — | — | — |
| | opposite | — | — | — | — |
| LIBERO-Spatial | original | — | — | — | — |
| | empty | — | — | — | — |
| | shuffle | — | — | — | — |
| | random | — | — | — | — |
| | synonym | — | — | — | — |
| | opposite | — | — | — | — |
| LIBERO-Object | original | — | — | — | — |
| | empty | — | — | — | — |
| | shuffle | — | — | — | — |
| | random | — | — | — | — |
| | synonym | — | — | — | — |
| | opposite | — | — | — | — |
| LIBERO-Goal | original | — | — | — | — |
| | empty | — | — | — | — |
| | shuffle | — | — | — | — |
| | random | — | — | — | — |
| | synonym | — | — | — | — |
| | opposite | — | — | — | — |

### Attention IoU (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos |
|---|---|---|---|---|
| LIBERO-10 | original | — | — | — |
| | empty | — | — | — |
| | shuffle | — | — | — |
| | random | — | — | — |
| | synonym | — | — | — |
| | opposite | — | — | — |
| LIBERO-Spatial | original | — | — | — |
| | empty | — | — | — |
| | shuffle | — | — | — |
| | random | — | — | — |
| | synonym | — | — | — |
| | opposite | — | — | — |
| LIBERO-Object | original | — | — | — |
| | empty | — | — | — |
| | shuffle | — | — | — |
| | random | — | — | — |
| | synonym | — | — | — |
| | opposite | — | — | — |
| LIBERO-Goal | original | — | — | — |
| | empty | — | — | — |
| | shuffle | — | — | — |
| | random | — | — | — |
| | synonym | — | — | — |
| | opposite | — | — | — |

### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos |
|---|---|---|---|---|
| LIBERO-10 | original | — | — | — |
| | empty | — | — | — |
| | shuffle | — | — | — |
| | random | — | — | — |
| | synonym | — | — | — |
| | opposite | — | — | — |
| LIBERO-Spatial | original | — | — | — |
| | empty | — | — | — |
| | shuffle | — | — | — |
| | random | — | — | — |
| | synonym | — | — | — |
| | opposite | — | — | — |
| LIBERO-Object | original | — | — | — |
| | empty | — | — | — |
| | shuffle | — | — | — |
| | random | — | — | — |
| | synonym | — | — | — |
| | opposite | — | — | — |
| LIBERO-Goal | original | — | — | — |
| | empty | — | — | — |
| | shuffle | — | — | — |
| | random | — | — | — |
| | synonym | — | — | — |
| | opposite | — | — | — |

---

## 2. Visual Perturbation

Conditions: **original** | **rotate 30°** | **translate 20% right** | **rotate 15° + translate 10%**

### Success Rate (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP |
|---|---|---|---|---|---|
| LIBERO-10 | original | — | — | — | — |
| | rotate 30° | — | — | — | — |
| | translate 20% | — | — | — | — |
| | rotate+translate | — | — | — | — |
| LIBERO-Spatial | original | — | — | — | — |
| | rotate 30° | — | — | — | — |
| | translate 20% | — | — | — | — |
| | rotate+translate | — | — | — | — |
| LIBERO-Object | original | — | — | — | — |
| | rotate 30° | — | — | — | — |
| | translate 20% | — | — | — | — |
| | rotate+translate | — | — | — | — |
| LIBERO-Goal | original | — | — | — | — |
| | rotate 30° | — | — | — | — |
| | translate 20% | — | — | — | — |
| | rotate+translate | — | — | — | — |
| LIBERO-90-Object | original | — | — | — | — |
| | rotate 30° | — | — | — | — |
| | translate 20% | — | — | — | — |
| | rotate+translate | — | — | — | — |
| LIBERO-90-Spatial | original | — | — | — | — |
| | rotate 30° | — | — | — | — |
| | translate 20% | — | — | — | — |
| | rotate+translate | — | — | — | — |
| LIBERO-90-Act | original | — | — | — | — |
| | rotate 30° | — | — | — | — |
| | translate 20% | — | — | — | — |
| | rotate+translate | — | — | — | — |
| LIBERO-90-Com | original | — | — | — | — |
| | rotate 30° | — | — | — | — |
| | translate 20% | — | — | — | — |
| | rotate+translate | — | — | — | — |

### Action Entropy (↓)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP |
|---|---|---|---|---|---|
| LIBERO-10 | original | — | — | — | — |
| | rotate 30° | — | — | — | — |
| | translate 20% | — | — | — | — |
| | rotate+translate | — | — | — | — |
| LIBERO-Spatial | original | — | — | — | — |
| | rotate 30° | — | — | — | — |
| | translate 20% | — | — | — | — |
| | rotate+translate | — | — | — | — |
| LIBERO-Object | original | — | — | — | — |
| | rotate 30° | — | — | — | — |
| | translate 20% | — | — | — | — |
| | rotate+translate | — | — | — | — |
| LIBERO-Goal | original | — | — | — | — |
| | rotate 30° | — | — | — | — |
| | translate 20% | — | — | — | — |
| | rotate+translate | — | — | — | — |

### Attention IoU (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos |
|---|---|---|---|---|
| LIBERO-10 | original | — | — | — |
| | rotate 30° | — | — | — |
| | translate 20% | — | — | — |
| | rotate+translate | — | — | — |
| LIBERO-Spatial | original | — | — | — |
| | rotate 30° | — | — | — |
| | translate 20% | — | — | — |
| | rotate+translate | — | — | — |
| LIBERO-Object | original | — | — | — |
| | rotate 30° | — | — | — |
| | translate 20% | — | — | — |
| | rotate+translate | — | — | — |
| LIBERO-Goal | original | — | — | — |
| | rotate 30° | — | — | — |
| | translate 20% | — | — | — |
| | rotate+translate | — | — | — |

### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos |
|---|---|---|---|---|
| LIBERO-10 | original | — | — | — |
| | rotate 30° | — | — | — |
| | translate 20% | — | — | — |
| | rotate+translate | — | — | — |
| LIBERO-Spatial | original | — | — | — |
| | rotate 30° | — | — | — |
| | translate 20% | — | — | — |
| | rotate+translate | — | — | — |
| LIBERO-Object | original | — | — | — |
| | rotate 30° | — | — | — |
| | translate 20% | — | — | — |
| | rotate+translate | — | — | — |
| LIBERO-Goal | original | — | — | — |
| | rotate 30° | — | — | — |
| | translate 20% | — | — | — |
| | rotate+translate | — | — | — |

---

## 3. Policy Perturbation

Conditions: **original** | **random action 25%** (random action replacement, p=0.25) | **object shift x** (Gaussian shift on object x-position, σ=5cm)

### Success Rate (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP |
|---|---|---|---|---|---|
| LIBERO-10 | original | — | — | — | — |
| | random action 25% | — | — | — | — |
| | object shift x | — | — | — | — |
| LIBERO-Spatial | original | — | — | — | — |
| | random action 25% | — | — | — | — |
| | object shift x | — | — | — | — |
| LIBERO-Object | original | — | — | — | — |
| | random action 25% | — | — | — | — |
| | object shift x | — | — | — | — |
| LIBERO-Goal | original | — | — | — | — |
| | random action 25% | — | — | — | — |
| | object shift x | — | — | — | — |
| LIBERO-90-Object | original | — | — | — | — |
| | random action 25% | — | — | — | — |
| | object shift x | — | — | — | — |
| LIBERO-90-Spatial | original | — | — | — | — |
| | random action 25% | — | — | — | — |
| | object shift x | — | — | — | — |
| LIBERO-90-Act | original | — | — | — | — |
| | random action 25% | — | — | — | — |
| | object shift x | — | — | — | — |
| LIBERO-90-Com | original | — | — | — | — |
| | random action 25% | — | — | — | — |
| | object shift x | — | — | — | — |

### Action Entropy (↓)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos | DP |
|---|---|---|---|---|---|
| LIBERO-10 | original | — | — | — | — |
| | random action 25% | — | — | — | — |
| | object shift x | — | — | — | — |
| LIBERO-Spatial | original | — | — | — | — |
| | random action 25% | — | — | — | — |
| | object shift x | — | — | — | — |
| LIBERO-Object | original | — | — | — | — |
| | random action 25% | — | — | — | — |
| | object shift x | — | — | — | — |
| LIBERO-Goal | original | — | — | — | — |
| | random action 25% | — | — | — | — |
| | object shift x | — | — | — | — |

### Attention IoU (↑)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos |
|---|---|---|---|---|
| LIBERO-10 | original | — | — | — |
| | random action 25% | — | — | — |
| | object shift x | — | — | — |
| LIBERO-Spatial | original | — | — | — |
| | random action 25% | — | — | — |
| | object shift x | — | — | — |
| LIBERO-Object | original | — | — | — |
| | random action 25% | — | — | — |
| | object shift x | — | — | — |
| LIBERO-Goal | original | — | — | — |
| | random action 25% | — | — | — |
| | object shift x | — | — | — |

### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 | OpenVLA | Cosmos |
|---|---|---|---|---|
| LIBERO-10 | original | — | — | — |
| | random action 25% | — | — | — |
| | object shift x | — | — | — |
| LIBERO-Spatial | original | — | — | — |
| | random action 25% | — | — | — |
| | object shift x | — | — | — |
| LIBERO-Object | original | — | — | — |
| | random action 25% | — | — | — |
| | object shift x | — | — | — |
| LIBERO-Goal | original | — | — | — |
| | random action 25% | — | — | — |
| | object shift x | — | — | — |
