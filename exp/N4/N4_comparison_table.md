# N4 Mixed-Supervision Experiment Comparison

Source files: `exp/N4/*_train_log.csv` and `exp/N4/*_console_output.log`

Selection metric:

`composite = 0.5 * val_lesion_dice + 0.25 * val_patient_bacc + 0.25 * val_region_bacc`

All values below are taken from each run's best-composite validation epoch.

## Main Comparison

| Rank | Run | Epochs | Best Epoch | Composite | Dice | Lesion Sens | Lesion Spec | Patient Sens | Patient Spec | Region Sens | Region Spec |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | M05_EM10_Curr10_30_ClampN2P2_P085N010 | 80 | 60 | 0.4849 | 0.3398 | 0.4380 | 0.9927 | 0.7833 | 0.6095 | 0.1727 | 0.9549 |
| 2 | M02_FixedW010_025_Curr10_20_P09N005 | 67 | 47 | 0.4834 | 0.3400 | 0.4762 | 0.9911 | 0.7944 | 0.6476 | 0.0818 | 0.9832 |
| 3 | M01_FixedW025_050_Curr10_30_P09N005 | 49 | 29 | 0.4775 | 0.3373 | 0.3869 | 0.9946 | 0.7111 | 0.7429 | 0.0455 | 0.9712 |
| 4 | M03_EM1_Curr10_30_ClampN05P1_P09N005 | 63 | 43 | 0.4756 | 0.3307 | 0.4090 | 0.9934 | 0.7778 | 0.5905 | 0.1455 | 0.9681 |
| 5 | M06_EM1_NoCurr_ClampN05P1_P09N005 | 58 | 38 | 0.4748 | 0.3166 | 0.4229 | 0.9916 | 0.8556 | 0.5333 | 0.1909 | 0.9519 |
| 6 | M04_EM5_Curr10_20_ClampN1P2_P09N005 | 43 | 23 | 0.4742 | 0.3229 | 0.4783 | 0.9890 | 0.5500 | 0.9429 | 0.0091 | 1.0000 |

## Metric-Focused Notes

- Best composite: M05, with 0.4849.
- Best Dice: M02, with 0.3400.
- Best lesion sensitivity: M04, with 0.4783.
- Best lesion specificity: M01, with 0.9946.
- Best patient sensitivity: M06, with 0.8556.
- Best patient specificity: M04, with 0.9429.
- Best region sensitivity: M06, with 0.1909.
- Best region specificity: M04, with 1.0000.
