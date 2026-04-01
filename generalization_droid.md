# Generalization Experiments

Models: **pi0.5** | **DP** (diffusion policy, trained from scratch)
Suites: -In domain | -Spatial | -Object | LIBERO-Goal
Metrics: success rate (↑), action entropy (↓), attention IoU (↑), attention ratio (visual/total)
*Attention metrics apply to VLA models only (pi0.5).*

---

## 1. Language Perturbation


### Success Rate (↑)

| Suite | Condition | pi0.5 | Dreamzero |
|---|---|---|---|
| DROID-Object | original | 52% |
| | empty | 3% |
| | shuffle | 28% |
| | random | 2% |
| | synonym | 47% |
| DROID-Spatial | original | 22% |
| | empty | 0% |
| | shuffle | 8% |
| | random | 0% |
| | synonym | 18% |
| DROID-Act | original | 48% |
| | empty | 2% |
| | shuffle | 33% |
| | random | 0% |
| | synonym | 52% |
| DROID-Com | original | 0% |
| | empty | 0% |
| | shuffle | 0% |
| | random | 0% |
| | synonym | 2% |

<!-- ### Action Entropy (↓)

| Suite | Condition | pi0.5 | DP |
|---|---|---|
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| DROID-Spatial | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| DROID-Act | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| DROID-Com | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — | -->

### Attention IoU (↑)

| Suite | Condition | pi0.5 | Dreamzero |
|---|---|---|
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| DROID-Spatial | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| DROID-Act | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| DROID-Com | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |

### Attention Ratio (visual / total)

| Suite | Condition | pi0.5 | Dreamzero |
|---|---|---|
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| DROID-Spatial | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
| DROID-Goal | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |

---

## 2. Visual Perturbation

Conditions: **original** | **rotate 30°** | **translate 20% right** | **rotate 15° + translate 10%**

### Success Rate (↑)

| Suite | Condition | pi0.5 | Dreamzero |
|---|---|---|
| DROID-Object | original | 52% |
| | rotate 30° | 2% |
| | translate 20% | 47% |
| | rotate+translate | 0% |
| DROID-Spatial | original | 22% |
| | rotate 30° | 0% |
| | translate 20% | 18% |
| | rotate+translate | 0% |
| DROID-Act | original | 48% |
| | rotate 30° | 3% |
| | translate 20% | 52% |
| | rotate+translate | 2% |
| DROID-Com | original | 0% |
| | rotate 30° | 0% |
| | translate 20% | 0% |
| | rotate+translate | 0% |

<!-- ### Action Entropy (↓)

| Suite | Condition | pi0.5 | DP |
|---|---|---|
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
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
| | rotate+translate | — | -->

### Attention IoU (↑)

| Suite | Condition | pi0.5 |
|---|---|---|
| DROID-Object | original | — |
| | empty | — |
| | shuffle | — |
| | random | — |
| | synonym | — |
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

| Suite | Condition | pi0.5 | Dreamzero |
|---|---|---|---|
| DROID-Object | original | 52% |
| | random action 25% | 3% |
| | object shift x | 45% |
| DROID-Spatial | original | 22% |
| | random action 25% | 0% |
| | object shift x | 17% |
| DROID-Act | original | 48% |
| | random action 25% | 2% |
| | object shift x | 48% |
| DROID-Com | original | 0% |
| | random action 25% | 0% |
| | object shift x | 0% |


### Attention IoU (↑)

| Suite | Condition | pi0.5 | Dreamzero | 
|---|---|---|---|
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

| Suite | Condition | pi0.5 | Dreamzero |
|---|---|---|---|
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
