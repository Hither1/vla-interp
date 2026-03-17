"""Temporal reliance comparison: instantaneous vs temporally integrated grounding.

Core question
─────────
Is success governed more by current-frame grounding (IoU_t) or by temporally
accumulated state/history (rolling mean IoU_[t-k:t])?

Predictions
───────────
pi0.5  (flow-matching VLA, no temporal memory):
  current-step IoU should be more predictive; instantaneous failures hurt.

DreamZero  (world-model-like, temporal KV-cache):
  rolling / cumulative IoU should predict success better; brief failures tolerated.

Metric: Temporal Integration Score (TIS)
  TIS = mean_t|r(success, cumul_IoU_t)| / mean_t|r(success, instant_IoU_t)|
  DreamZero > pi0.5 is the predicted finding.

Plots
─────
  T1  Predictive power (Pearson r) vs episode progress —
      instantaneous vs cumulative mean, per model
  T2  TIS bar chart
  T3  Within-episode IoU trajectory: success vs failure, per model
  T4  Predictive r vs rolling window k
  T5  Early-half vs late-half IoU scatter, colored by success
  T6  2x2 summary combining T1, T3, T4, TIS bar

Data sources
────────────
DreamZero: data/libero/dreamzero/perturb/none/{suite}/*.json
           "attention_steps": [{t, iou_iou, visual_linguistic_ratio}, ...]

pi0.5:     data/libero/pi0.5/perturb/none/{suite}/*.json  (native format)
           OR --pi05-iou-dir pointing at evaluate_attention_iou.py output dir
              containing iou_results_{suite}.json files
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# paths & style
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "libero"
OUT_DIR   = Path(__file__).resolve().parent / "plots_temporal"
OUT_DIR.mkdir(exist_ok=True)

MODEL_COLORS = {"pi0.5": "#4C72B0", "DreamZero": "#17becf"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
})


# ===============================================================
# DATA LOADING
# ===============================================================

