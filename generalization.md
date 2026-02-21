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
