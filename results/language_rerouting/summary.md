# Language-Conditioning Comparison

This summary compares `prompt_mode=original` against language-corrupted prompts.

## Coverage

- DreamZero: 1320 discovered episode records
- paired clean/corrupt comparisons: 1100

## Group Means

- DreamZero / empty: delta_ratio=0.011541679909469835, delta_iou=-0.0034021912913870574, delta_success=-0.06363636363636363, VCI=0.011541679909469835, GRI=-0.031034564031174484
- DreamZero / opposite: delta_ratio=0.010483013854850298, delta_iou=-0.011107815037606656, delta_success=-0.022727272727272728, VCI=0.010483013854850298, GRI=-0.04349865733009367
- DreamZero / random: delta_ratio=0.011814449849601486, delta_iou=-0.008647431975434331, delta_success=-0.08181818181818182, VCI=0.011814449849601486, GRI=-0.038511669195345356
- DreamZero / shuffle: delta_ratio=0.0001434683307697809, delta_iou=-0.006613042340623137, delta_success=-0.022727272727272728, VCI=0.0001434683307697809, GRI=-0.04688508468218869
- DreamZero / synonym: delta_ratio=-0.010161929420681875, delta_iou=0.0015392952109757143, delta_success=0.00909090909090909, VCI=-0.010161929420681875, GRI=-0.059083267577612404
