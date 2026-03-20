# Language-Conditioning Comparison

This summary compares `prompt_mode=original` against language-corrupted prompts.

## Coverage

- DreamZero: 1080 discovered episode records
- pi0.5: 207 discovered episode records
- paired clean/corrupt comparisons: 1000

## Group Means

- DreamZero / empty: delta_ratio=0.010646729574297232, delta_iou=None, delta_success=-0.03888888888888889, VCI=0.010646729574297232, GRI=None
- DreamZero / opposite: delta_ratio=0.009659229837074151, delta_iou=None, delta_success=0.0, VCI=0.009659229837074151, GRI=None
- DreamZero / random: delta_ratio=0.010100358211424066, delta_iou=None, delta_success=-0.03333333333333333, VCI=0.010100358211424066, GRI=None
- DreamZero / shuffle: delta_ratio=-0.0005646130342349077, delta_iou=None, delta_success=-0.011111111111111112, VCI=-0.0005646130342349077, GRI=None
- DreamZero / synonym: delta_ratio=-0.010769425792638029, delta_iou=None, delta_success=0.011111111111111112, VCI=-0.010769425792638029, GRI=None
- pi0.5 / empty: delta_ratio=0.0037499597689451786, delta_iou=None, delta_success=0.0, VCI=0.0037499597689451786, GRI=None
- pi0.5 / opposite: delta_ratio=-0.022561616901354282, delta_iou=None, delta_success=0.0, VCI=-0.022561616901354282, GRI=None
- pi0.5 / shuffle: delta_ratio=-0.01472040154827293, delta_iou=None, delta_success=0.0, VCI=-0.01472040154827293, GRI=None
- pi0.5 / synonym: delta_ratio=-0.01736701949435327, delta_iou=None, delta_success=0.0, VCI=-0.01736701949435327, GRI=None
