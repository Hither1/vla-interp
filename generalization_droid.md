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
| DROID-Object | original | 52% | 85% |
| | empty | 3% | 3% |
| | shuffle | 28% | 78% |
| | random | 2% | 2% |
| | synonym | 47% | 82% |
| DROID-Spatial | original | 22% | 73% |
| | empty | 0% | 2% |
| | shuffle | 8% | 65% |
| | random | 0% | 2% |
| | synonym | 18% | 70% |
| DROID-Act | original | 48% | 80% |
| | empty | 2% | 2% |
| | shuffle | 33% | 72% |
| | random | 0% | 2% |
| | synonym | 52% | 78% |
| DROID-Com | original | 0% | 0% |
| | empty | 0% | 0% |
| | shuffle | 0% | 0% |
| | random | 0% | 0% |
| | synonym | 0% | 0% |

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
|---|---|---|---|
| DROID-Object | original | 52% | 85% |
| | rotate 30° | 2% | 8% |
| | translate 20% | 45% | 48% |
| | rotate+translate | 0% | 5% |
| DROID-Spatial | original | 22% | 73% |
| | rotate 30° | 0% | 5% |
| | translate 20% | 18% | 38% |
| | rotate+translate | 0% | 3% |
| DROID-Act | original | 48% | 80% |
| | rotate 30° | 3% | 8% |
| | translate 20% | 42% | 45% |
| | rotate+translate | 2% | 5% |
| DROID-Com | original | 0% | 0% |
| | rotate 30° | 0% | 0% |
| | translate 20% | 0% | 0% |
| | rotate+translate | 0% | 0% |

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
| DROID-Object | original | 52% | 85% |
| | random action 25% | 3% | 22% |
| | object shift x | 45% | 72% |
| DROID-Spatial | original | 22% | 73% |
| | random action 25% | 0% | 18% |
| | object shift x | 17% | 62% |
| DROID-Act | original | 48% | 80% |
| | random action 25% | 2% | 20% |
| | object shift x | 48% | 68% |
| DROID-Com | original | 0% | 0% |
| | random action 25% | 0% | 0% |
| | object shift x | 0% | 0% |


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
