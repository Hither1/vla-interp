# Generalization Experiments

Models: **pi0.5** | **DP** (diffusion policy, trained from scratch)
Suites: -In domain | -Spatial | -Object | LIBERO-Goal
Metrics: success rate (↑), action entropy (↓), attention IoU (↑), attention ratio (visual/total)
*Attention metrics apply to VLA models only (pi0.5).*

---

## 1. Language Perturbation

Conditions: **original** | **empty** (no prompt) | **shuffle** (shuffled words) | **random** (random words) | **synonym** | **opposite**

### Success Rate (↑)

| Suite | Condition | pi0.5 | DP |
|---|---|---|
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Spatial | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Act | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Com | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |

### Action Entropy (↓)

| Suite | Condition | pi0.5 | DP |
|---|---|---|
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Spatial | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Act | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Com | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |

### Attention IoU (↑)

| Suite | Condition | pi0.5 |
|---|---|---|
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Spatial | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Act | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Com | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |

### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 |
|---|---|---|
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Spatial | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Goal | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |

---

## 2. Visual Perturbation

Conditions: **original** | **rotate 30°** | **translate 20% right** | **rotate 15° + translate 10%**

### Success Rate (↑)

| Suite | Condition | pi0.5 | DP |
|---|---|---|
| DROID-Object | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |
| DROID-Spatial | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |
| DROID-Act | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |
| DROID-Com | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |

### Action Entropy (↓)

| Suite | Condition | pi0.5 | DP |
|---|---|---|
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Spatial | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |
| DROID-Act | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |
| DROID-Com | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |

### Attention IoU (↑)

| Suite | Condition | pi0.5 |
|---|---|---|
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Spatial | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |
| DROID-Object | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |
| DROID-Goal | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |

### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 |
|---|---|---|
| DROID-Object | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |
| DROID-Spatial | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |
| DROID-Act | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |
| DROID-Com | original | — |
| | rotate 30° | — |
| | translate 20% | — |
| | rotate+translate | — |

---

## 3. Policy Perturbation

Conditions: **original** | **random action 25%** (random action replacement, p=0.25) | **object shift x** (Gaussian shift on object x-position, σ=5cm)

### Success Rate (↑)

| Suite | Condition | pi0.5 | DP |
|---|---|---|
| DROID-Object | original | — |
| | random action 25% | — |
| | object shift x | — |
| DROID-Spatial | original | — |
| | random action 25% | — |
| | object shift x | — |
| DROID-Act | original | — |
| | random action 25% | — |
| | object shift x | — |
| DROID-Com | original | — |
| | random action 25% | — |
| | object shift x | — |

### Action Entropy (↓)

| Suite | Condition | pi0.5 | DP |
|---|---|---|
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| | opposite | — |
| DROID-Spatial | original | — |
| | random action 25% | — |
| | object shift x | — |
| DROID-Object | original | — |
| | random action 25% | — |
| | object shift x | — |
| DROID-Goal | original | — |
| | random action 25% | — |
| | object shift x | — |

### Attention IoU (↑)

| Suite | Condition | pi0.5 |
|---|---|---|
| DROID-Object | original | — |
| | random action 25% | — |
| | object shift x | — |
| DROID-Spatial | original | — |
| | random action 25% | — |
| | object shift x | — |
| DROID-Act | original | — |
| | random action 25% | — |
| | object shift x | — |
| DROID-Com | original | — |
| | random action 25% | — |
| | object shift x | — |

### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 |
|---|---|---|
| DROID-Object | original | — |
| | random action 25% | — |
| | object shift x | — |
| DROID-Spatial | original | — |
| | random action 25% | — |
| | object shift x | — |
| DROID-Act | original | — |
| | random action 25% | — |
| | object shift x | — |
| DROID-Com | original | — |
| | random action 25% | — |
| | object shift x | — |
