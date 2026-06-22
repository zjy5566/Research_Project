# N2 Experiment Comparison

Composite = 0.50 * val_lesion_dice + 0.25 * val_patient_bacc + 0.25 * val_region_bacc.

## Best Composite Summary

| Rank | Experiment | Epochs | Config | Best composite | Dice | Patient BAcc | Region BAcc | Val loss |
|---:|---|---:|---|---:|---:|---:|---:|---:|
| 1 | N2_CM05_FixedW005_Curr15_P09N005 | 1-53 | EM=False, Curr=True, Start=15, W=0.05, Soft=0.9/0.05, EMlr=1.0, Clamp=False | 0.4826 @ 33 | 0.3454 | 0.7290 | 0.5104 | 0.2668 |
| 2 | N2_CM03_FixedW010_Curr10_P09N005 | 1-53 | EM=False, Curr=True, Start=10, W=0.1, Soft=0.9/0.05, EMlr=1.0, Clamp=False | 0.4780 @ 33 | 0.3474 | 0.6944 | 0.5225 | 0.2964 |
| 3 | N2_CM04_FixedW010_Curr15_P09N005 | 1-51 | EM=False, Curr=True, Start=15, W=0.1, Soft=0.9/0.05, EMlr=1.0, Clamp=False | 0.4752 @ 31 | 0.3447 | 0.6940 | 0.5174 | 0.3001 |
| 4 | N2_CM02_FixedW025_Curr10_P09N005 | 1-51 | EM=False, Curr=True, Start=10, W=0.25, Soft=0.9/0.05, EMlr=1.0, Clamp=False | 0.4746 @ 31 | 0.3351 | 0.7103 | 0.5180 | 0.3928 |
| 5 | N2_CM06_FixedW010_Curr10_P085N010 | 1-54 | EM=False, Curr=True, Start=10, W=0.1, Soft=0.85/0.1, EMlr=1.0, Clamp=False | 0.4709 @ 34 | 0.3344 | 0.6968 | 0.5180 | 0.3047 |
| 6 | N2_EM | 1-39 | EM=True, Curr=True, Start=10, W=0.25, Soft=0.9/0.05, EMlr=10.0, Clamp=True | 0.4669 @ 19 | 0.3267 | 0.7016 | 0.5128 | 0.8415 |
| 7 | N2_CM01_FixedW025_NoCurr_P09N005 | 1-65 | EM=False, Curr=False, Start=1, W=0.25, Soft=0.9/0.05, EMlr=1.0, Clamp=False | 0.4662 @ 45 | 0.3281 | 0.6218 | 0.5866 | 0.4125 |
| 8 | N2_CM07_EM1_Curr15_ClampN05P1_P09N005 | 1-62 | EM=True, Curr=True, Start=15, W=0.25, Soft=0.9/0.05, EMlr=1.0, Clamp=True | 0.4661 @ 42 | 0.3279 | 0.6302 | 0.5782 | 0.8486 |
| 9 | N2_0.25weight_softlabel | 1-65 | EM=False, Curr=False, Start=1, W=0.25, Soft=0.9/0.05, EMlr=10.0, Clamp=False | 0.4648 @ 45 | 0.3281 | 0.6218 | 0.5812 | 0.4101 |
| 10 | N2_lower_weight | 1-65 | EM=False, Curr=False, Start=1, W=0.25, Soft=/, EMlr=10.0, Clamp=False | 0.4534 @ 45 | 0.3318 | 0.5405 | 0.6096 | 0.4066 |
| 11 | N2_Soft_label | 1-56 | EM=False, Curr=False, Start=1, W=1.0, Soft=0.9/0.05, EMlr=10.0, Clamp=False | 0.4501 @ 36 | 0.3244 | 0.6361 | 0.5156 | 0.8641 |
| 12 | N2 | 1-3 | EM=False, Curr=False, Start=1, W=1.0, Soft=/, EMlr=10.0, Clamp=False | 0.2840 @ 3 | 0.0332 | 0.5694 | 0.5000 | 0.9344 |

## Key Metric Best Epochs

| Experiment | Dice max | F1 max | Patient BAcc max | Patient AUC max | Region BAcc max | Region AUC max |
|---|---:|---:|---:|---:|---:|---:|
| N2_CM05_FixedW005_Curr15_P09N005 | 0.3521 @ 41 | 0.4062 @ 49 | 0.7290 @ 33 | 0.7949 @ 16 | 0.5901 @ 12 | 0.6467 @ 16 |
| N2_CM03_FixedW010_Curr10_P09N005 | 0.3474 @ 33 | 0.3906 @ 30 | 0.7103 @ 14 | 0.7944 @ 36 | 0.5633 @ 9 | 0.6458 @ 18 |
| N2_CM04_FixedW010_Curr15_P09N005 | 0.3486 @ 44 | 0.3974 @ 44 | 0.7373 @ 39 | 0.7982 @ 27 | 0.5901 @ 12 | 0.6501 @ 15 |
| N2_CM02_FixedW025_Curr10_P09N005 | 0.3517 @ 35 | 0.3941 @ 35 | 0.7175 @ 39 | 0.8001 @ 27 | 0.5633 @ 9 | 0.6594 @ 19 |
| N2_CM06_FixedW010_Curr10_P085N010 | 0.3467 @ 35 | 0.3905 @ 35 | 0.6968 @ 34 | 0.7936 @ 11 | 0.5633 @ 9 | 0.6474 @ 18 |
| N2_EM | 0.3287 @ 36 | 0.3709 @ 30 | 0.7016 @ 19 | 0.7920 @ 29 | 0.5839 @ 37 | 0.6486 @ 16 |
| N2_CM01_FixedW025_NoCurr_P09N005 | 0.3365 @ 46 | 0.3842 @ 46 | 0.7008 @ 4 | 0.7926 @ 53 | 0.5925 @ 52 | 0.6260 @ 24 |
| N2_CM07_EM1_Curr15_ClampN05P1_P09N005 | 0.3397 @ 35 | 0.3863 @ 46 | 0.7091 @ 1 | 0.7959 @ 29 | 0.6045 @ 46 | 0.6461 @ 22 |
| N2_0.25weight_softlabel | 0.3365 @ 46 | 0.3842 @ 46 | 0.7008 @ 4 | 0.7926 @ 53 | 0.5839 @ 52 | 0.6126 @ 24 |
| N2_lower_weight | 0.3390 @ 35 | 0.3824 @ 45 | 0.6940 @ 4 | 0.7813 @ 31 | 0.6181 @ 59 | 0.6242 @ 54 |
| N2_Soft_label | 0.3294 @ 35 | 0.3718 @ 45 | 0.6742 @ 20 | 0.7937 @ 36 | 0.6006 @ 37 | 0.6401 @ 24 |
| N2 | 0.0332 @ 3 | 0.0459 @ 3 | 0.5694 @ 3 | 0.6826 @ 3 | 0.5000 @ 1 | 0.5723 @ 3 |
