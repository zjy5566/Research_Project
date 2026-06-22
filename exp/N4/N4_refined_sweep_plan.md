# N4 Refined Sweep Plan

Selection metric:

`composite = 0.50 * val_lesion_dice + 0.25 * val_patient_bacc + 0.25 * val_region_bacc`

## Evidence From Previous Runs

| Source | Best setting | Best epoch | Composite | Dice | Patient BAcc | Region BAcc | Takeaway |
|---|---|---:|---:|---:|---:|---:|---|
| N2 TBx-only | `FixedW005_Curr15_P09N005` | 33 | 0.4826 | 0.3454 | 0.7290 | 0.5104 | TBx works best with low weight `0.05` and late start `15`. |
| N3 SBx-only | `FixedSys025_Start15` | 33 | 0.4833 | 0.3412 | 0.7282 | 0.5225 | SBx works best with fixed weight `0.25` and start `15`. |
| N4 previous fixed | `FixedW010_025_Curr10_20_P09N005` | 47 | 0.4834 | 0.3400 | 0.7210 | 0.5325 | Best fixed N4 used SBx weight `0.25`, but TBx weight may still be too high. |
| N4 previous overall | `EM10_Curr10_30_ClampN2P2_P085N010` | 60 | 0.4849 | 0.3398 | 0.6964 | 0.5638 | EM10 gives best composite, mostly through higher region BAcc. |

## Refined N4 Experiments

Main hypothesis: keep SBx fixed weight at `0.25`, reduce TBx fixed weight to the N2 optimum `0.05`, and sweep the time at which TBx/SBx are added.

| Run | Weighting | TBx start | SBx start | TBx weight | SBx weight | Soft label | Purpose |
|---|---|---:|---:|---:|---:|---|---|
| R01 | Fixed | 15 | 15 | 0.05 | 0.25 | 0.9/0.05 | Directly combine the best N2 and N3 single-task settings. |
| R02 | Fixed | 15 | 20 | 0.05 | 0.25 | 0.9/0.05 | Let TBx enter first, then add SBx later. |
| R03 | Fixed | 10 | 15 | 0.05 | 0.25 | 0.9/0.05 | Earlier TBx, SBx at the N3-best start time. |
| R04 | Fixed | 10 | 20 | 0.05 | 0.25 | 0.9/0.05 | Same timing as previous best fixed N4, but lower TBx weight. |
| R05 | Fixed | 15 | 15 | 0.10 | 0.25 | 0.9/0.05 | Test whether previous N4 still wants higher TBx weight when both tasks start late. |
| R06 | Fixed | 10 | 20 | 0.10 | 0.25 | 0.9/0.05 | Re-run previous best fixed timing/weights as a direct anchor. |
| R07 | Fixed | 15 | 15 | 0.05 | 0.50 | 0.9/0.05 | Test whether stronger SBx helps region BAcc or hurts Dice. |
| R08 | Fixed | 1 | 1 | 0.05 | 0.25 | 0.9/0.05 | No-curriculum control with tuned fixed weights. |
| R09 | EM10 | 15 | 15 | ignored | ignored | 0.9/0.05 | Compare EM10 with later starts and N2-best soft labels. |
| R10 | EM10 | 15 | 20 | ignored | ignored | 0.85/0.10 | Compare against previous best EM soft-label style with later starts. |
| R11 | Fixed | 15 | 30 | 0.05 | 0.25 | 0.9/0.05 | Combine N2-best TBx timing with the previous best N4 late SBx timing. |
| R12 | EM10 | 15 | 30 | ignored | ignored | 0.85/0.10 | Directly test whether delaying TBx to 15 improves the previous best EM10 10/30 pattern. |

All runs keep `MASK_TARGET_IN_SYS=True`, so TBx voxels are removed from SBx zone masks when both labels are present.
