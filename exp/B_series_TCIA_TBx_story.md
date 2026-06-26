# B-series experiment story after 2026-06-23

## Motivation

The main baseline should no longer be PUB radiologist-only dense lesion
segmentation. The new baseline is TCIA target-biopsy-confirmed radiologist
target lesion ROI supervision:

- A radiologist first marks a suspicious MRI lesion.
- Target biopsy confirms the sampled target.
- Only the target ROI is treated as positive evidence when the biopsy-confirmed
  grade passes `LESION_POSITIVE_THRESHOLD`.
- Unsampled voxels and biopsy-negative/low-grade target ROIs are not used as
  background negatives in the TCIA TBx ROI loss.

This makes the baseline a clinically grounded weak-localisation setup rather
than a needle-track proxy or a pure public radiologist-mask baseline.

## Proposed B-series

| Mode | Training supervision | Purpose |
| --- | --- | --- |
| `B1_TCIA_TBX_BASELINE` | TCIA TBx-confirmed target ROI only | Main baseline. Tests whether biopsy-confirmed target ROIs can train a lesion risk map. |
| `B2_TCIA_SBX_ONLY` | TCIA SBx region labels only | Region-only weak supervision ablation. |
| `B3_TCIA_TBX_SBX` | TCIA TBx-confirmed target ROI + TCIA SBx region labels | Tests whether adding systematic-biopsy regional evidence improves clinical localisation. |
| Legacy N-series | PUB radiologist masks optionally mixed with TCIA | Kept only for backwards comparison with previous runs. |

## Loss definition

For TCIA TBx ROI supervision, the default loss is positive-only:

```text
L_TBx = mean_{v: target_mask(v) >= positive_threshold} -log p(v)
```

There is no `(1 - y) * log(1 - p)` term for unlabelled voxels or
biopsy-negative target ROIs. This avoids teaching the network that unsampled
or biopsy-negative regions are true voxel-level background.

SBx/PROMIS region supervision remains region-level MIL and can still use
sampled benign regions as valid negatives.

## PROMIS GT interpretation

PROMIS contains both radiologist MRI lesion information and template-prostate
mapping biopsy information. These are different label sources:

- Histopathology/template-biopsy labels define cancer-positive biopsy regions.
- Radiologist lesion contours/readings define MRI-suspicious regions.
- The PROMIS-curated localisation code converts radiologist lesion masks into
  zone-level MRI-positive labels by checking overlap/IoU between lesion masks
  and generated prostate zones.

Therefore a rule such as "lesion overlap with a region > 5%" is best understood
as a way to convert voxel-level radiologist lesion annotations into region-level
MRI-positive labels for localisation analysis. It is not the same thing as the
biopsy-derived histopathology GT for sampled regions.
