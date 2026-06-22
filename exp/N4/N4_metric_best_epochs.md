# N4 Metric Best-Epoch Comparison

数据来源: `exp/N4/*_train_log.csv`

口径:

- 不再使用 composite 选择 epoch。
- 每个指标单独选择该指标数值最高的 epoch。
- 指标保留 Dice、Sensitivity、Specificity。
- 若多个 epoch 并列，使用日志中最早出现的 epoch。

## Best Dice Epoch

按每个实验的最佳 lesion Dice epoch 排序，并报告该 epoch 下各任务的 sensitivity/specificity。

| Rank | Run | Best Dice Epoch | Dice | Lesion Sens | Lesion Spec | Patient Sens | Patient Spec | Region Sens | Region Spec |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | M02_FixedW010_025_Curr10_20_P09N005 | 44 | 0.3431 | 0.4583 | 0.9917 | 0.8444 | 0.3810 | 0.1182 | 0.9609 |
| 2 | M05_EM10_Curr10_30_ClampN2P2_P085N010 | 60 | 0.3398 | 0.4380 | 0.9927 | 0.7833 | 0.6095 | 0.1727 | 0.9549 |
| 3 | M03_EM1_Curr10_30_ClampN05P1_P09N005 | 41 | 0.3395 | 0.4886 | 0.9902 | 0.8778 | 0.3810 | 0.1818 | 0.9333 |
| 4 | M01_FixedW025_050_Curr10_30_P09N005 | 49 | 0.3390 | 0.4281 | 0.9934 | 0.8278 | 0.4190 | 0.1091 | 0.9483 |
| 5 | M04_EM5_Curr10_20_ClampN1P2_P09N005 | 25 | 0.3309 | 0.5216 | 0.9876 | 0.9500 | 0.2286 | 0.2182 | 0.9489 |
| 6 | M06_EM1_NoCurr_ClampN05P1_P09N005 | 43 | 0.3196 | 0.3870 | 0.9935 | 0.8944 | 0.2571 | 0.2818 | 0.8882 |

## Per-Run Best Epoch By Metric

每个单元格格式为 `E{epoch}: {best value}`。

| Run | Dice | Lesion Sens | Lesion Spec | Patient Sens | Patient Spec | Region Sens | Region Spec |
|---|---:|---:|---:|---:|---:|---:|---:|
| M01_FixedW025_050_Curr10_30_P09N005 | E49: 0.3390 | E16: 0.6060 | E32: 0.9954 | E2: 0.9944 | E32: 0.8857 | E8: 0.4182 | E35: 0.9964 |
| M02_FixedW010_025_Curr10_20_P09N005 | E44: 0.3431 | E16: 0.6017 | E32: 0.9962 | E2: 0.9944 | E21: 0.9524 | E8: 0.4182 | E32: 0.9970 |
| M03_EM1_Curr10_30_ClampN05P1_P09N005 | E41: 0.3395 | E16: 0.6024 | E32: 0.9969 | E2: 0.9944 | E12: 0.8857 | E6: 0.4364 | E32: 1.0000 |
| M04_EM5_Curr10_20_ClampN1P2_P09N005 | E25: 0.3309 | E14: 0.5703 | E32: 0.9970 | E6: 0.9944 | E23: 0.9429 | E6: 0.4182 | E23: 1.0000 |
| M05_EM10_Curr10_30_ClampN2P2_P085N010 | E60: 0.3398 | E14: 0.6031 | E32: 0.9972 | E2: 0.9944 | E32: 0.9810 | E6: 0.4091 | E19: 1.0000 |
| M06_EM1_NoCurr_ClampN05P1_P09N005 | E43: 0.3196 | E40: 0.5138 | E3: 1.0000 | E20: 0.9944 | E3: 1.0000 | E40: 0.3455 | E4: 1.0000 |

## Overall Best Metric Epochs

在所有 N4 实验和所有 epoch 中，为每个指标选择全局最高值，并报告同一 epoch 的 Dice/Sens/Spec。

| Metric | Best Run | Epoch | Value | Dice | Lesion Sens | Lesion Spec | Patient Sens | Patient Spec | Region Sens | Region Spec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Lesion Dice | M02_FixedW010_025_Curr10_20_P09N005 | 44 | 0.3431 | 0.3431 | 0.4583 | 0.9917 | 0.8444 | 0.3810 | 0.1182 | 0.9609 |
| Lesion Sensitivity | M01_FixedW025_050_Curr10_30_P09N005 | 16 | 0.6060 | 0.3014 | 0.6060 | 0.9799 | 0.8389 | 0.4476 | 0.0727 | 0.9555 |
| Lesion Specificity | M06_EM1_NoCurr_ClampN05P1_P09N005 | 3 | 1.0000 | 0.0005 | 0.0007 | 1.0000 | 0.0333 | 1.0000 | 0.0000 | 1.0000 |
| Patient Sensitivity | M04_EM5_Curr10_20_ClampN1P2_P09N005 | 6 | 0.9944 | 0.2701 | 0.4871 | 0.9821 | 0.9944 | 0.0000 | 0.4182 | 0.5919 |
| Patient Specificity | M06_EM1_NoCurr_ClampN05P1_P09N005 | 3 | 1.0000 | 0.0005 | 0.0007 | 1.0000 | 0.0333 | 1.0000 | 0.0000 | 1.0000 |
| Region Sensitivity | M03_EM1_Curr10_30_ClampN05P1_P09N005 | 6 | 0.4364 | 0.2735 | 0.4954 | 0.9819 | 0.9889 | 0.0095 | 0.4364 | 0.6088 |
| Region Specificity | M05_EM10_Curr10_30_ClampN2P2_P085N010 | 32 | 1.0000 | 0.2946 | 0.2721 | 0.9972 | 0.5222 | 0.9810 | 0.0000 | 1.0000 |

## Notes

- 以最佳 Dice 为主时，M02 最好，最佳 Dice 为 0.3431，出现在 epoch 44。
- M05 的最佳 Dice 排名第二，且 patient specificity 更高，但 lesion Dice 略低于 M02。
- 单独最大化 specificity 时会出现退化解。例如 M06 epoch 3 的 lesion/patient/region specificity 都很高，但 Dice 和 sensitivity 接近 0，不适合作为综合最优模型。
- Region sensitivity 的全局最高值来自 M03 epoch 6，但对应 region specificity 只有 0.6088，需要和具体任务目标一起判断。
