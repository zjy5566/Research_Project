# Method and Experiment Framework: Mixed Supervision for Prostate Lesion Risk Prediction

## 2. Method

### 2.1 Study Aim

This study investigates whether mixed supervision from different annotation granularities can improve prostate cancer lesion risk prediction on multiparametric MRI (mpMRI). Instead of relying on a single type of annotation, the proposed framework integrates three complementary supervision sources: voxel-level radiologist annotation (RA), target-ROI-level targeted biopsy supervision (TBx), and region-level systematic biopsy supervision (SBx).

The central hypothesis is that these supervision sources provide different but complementary information. RA provides detailed spatial information about lesion appearance and boundary. TBx provides pathology-confirmed evidence for radiologist-suspicious target lesions. SBx provides broader region-level histopathological information. By combining these labels, the model may learn a more clinically meaningful lesion risk map than models trained with a single supervision source.

### 2.2 Supervision Sources

The study uses three types of prostate cancer supervision:

| Supervision source | Spatial level | Description | Role in this study |
|---|---|---|---|
| RA | voxel-level | Dense radiologist-defined lesion masks | Provides spatial lesion shape and boundary information |
| TBx | target-ROI-level / lesion-level | Targeted-biopsy-confirmed suspicious lesion regions | Provides local pathology-confirmed lesion evidence |
| SBx | region-level | Systematic-biopsy labels assigned to prostate anatomical regions | Provides weak but clinically grounded regional evidence |

RA is treated as the strongest spatial supervision because it provides voxel-level lesion masks. However, it is radiologist-defined and may not always correspond directly to histopathological confirmation. TBx is more clinically specific because it is linked to targeted biopsy results, but it only supervises the sampled target lesion region rather than the full lesion extent. SBx provides histopathological labels over anatomical regions, but its spatial resolution is coarser than RA and TBx.

### 2.3 Overall Framework

The proposed model takes three-channel mpMRI as input, including T2-weighted imaging, diffusion-weighted imaging (DWI), and apparent diffusion coefficient (ADC). The network outputs a voxel-level lesion risk map, representing the predicted probability of clinically significant prostate cancer (csPCa) at each voxel.

All supervision sources are used to train the same lesion risk map. This is an important design choice: rather than building separate prediction heads for segmentation, target biopsy classification, and regional biopsy classification, the model learns one unified risk representation. Each supervision source constrains this risk map at a different spatial level.

The overall learning process can be summarised as:

```text
mpMRI input: T2 + DWI + ADC
        |
        v
3D lesion prediction network
        |
        v
voxel-level lesion risk map
        |
        +-- RA supervision: voxel-level lesion mask loss
        +-- TBx supervision: target-ROI-level biopsy-confirmed loss
        +-- SBx supervision: region-level MIL loss
```

### 2.4 Model Architecture

A 3D residual U-Net is used as the lesion prediction backbone. The encoder extracts hierarchical spatial features from mpMRI, and the decoder restores the spatial resolution to generate a voxel-level lesion risk map. Residual blocks are used to improve feature learning and training stability in 3D medical images.

The model has one primary output: lesion logits. These logits can be interpreted at different levels depending on the available supervision. For RA, the logits are compared directly with voxel-level lesion masks. For TBx, the loss is computed only within sampled target lesion regions. For SBx, lesion logits inside each anatomical region are pooled to obtain region-level predictions.

### 2.5 Mixed-Supervision Learning

The training objective is defined according to the available annotation type for each case.

For RA cases, dense voxel-level supervision is used to train the model to predict lesion shape and location. For TBx cases, only voxels inside targeted-biopsy-confirmed regions contribute to the loss. Unsampled voxels are not treated as negative background, because their true pathological status is unknown. For SBx cases, voxel-level predictions are aggregated into region-level scores using multiple instance learning (MIL), and these region scores are compared with systematic biopsy labels.

The overall loss can be written conceptually as:

```text
L_total =
    w_RA  * L_RA
  + w_TBx * L_TBx
  + w_SBx * L_SBx
```

where `L_RA` is the voxel-level radiologist annotation loss, `L_TBx` is the targeted-biopsy ROI loss, and `L_SBx` is the region-level systematic biopsy loss. Different experiments activate different combinations of these terms to evaluate the contribution of each supervision source.

## 3. Experiments

### 3.1 Experimental Aim

The experiments are designed to answer the following question:

> Does mixed supervision improve prostate lesion risk prediction compared with single-source supervision?

To answer this, the experiments compare models trained with individual supervision sources and their combinations. The goal is not only to identify the best-performing setting, but also to understand what each supervision source contributes to lesion localisation and clinical prediction.

### 3.2 Experimental Groups

The experiments can be organised into single-supervision and mixed-supervision settings.

#### Single-supervision baselines

| Experiment | Supervision | Purpose |
|---|---|---|
| RA only | voxel-level radiologist annotation | Tests how well dense radiologist masks train lesion localisation |
| TBx only | target-ROI-level biopsy-confirmed supervision | Tests whether targeted biopsy labels alone can train a lesion risk map |
| SBx only | region-level systematic biopsy labels | Tests whether weak regional pathology labels can provide useful supervision |

#### Mixed-supervision experiments

| Experiment | Supervision | Purpose |
|---|---|---|
| RA + TBx | voxel-level + target-ROI-level | Tests whether biopsy-confirmed target evidence improves RA-based learning |
| RA + SBx | voxel-level + region-level | Tests whether regional pathology labels improve dense lesion supervision |
| TBx + SBx | target-ROI-level + region-level | Tests whether two pathology-based supervision sources are complementary |
| RA + TBx + SBx | voxel-level + target-ROI-level + region-level | Tests the full mixed-supervision hypothesis |

This design allows the study to compare strong spatial supervision, local pathology-confirmed supervision, weak regional supervision, and their combinations.

### 3.3 Evaluation Strategy

Because the supervision sources operate at different spatial levels, the evaluation should also be reported at multiple levels.

#### Lesion-level / voxel-level evaluation

Voxel-level metrics are used when dense lesion masks are available. These metrics evaluate how well the predicted risk map matches radiologist-defined lesion locations.

Recommended metrics:

- Dice coefficient
- lesion sensitivity
- lesion specificity
- F1 score

#### Target-ROI-level evaluation

For TBx supervision, the model should also be evaluated within targeted biopsy regions. This is important because TBx does not provide complete dense lesion masks, but it provides clinically confirmed target regions.

Recommended metrics:

- TBx ROI AUC
- TBx ROI AUPRC
- TBx ROI balanced accuracy
- sensitivity at fixed specificity

#### Region-level evaluation

For SBx supervision, predictions should be aggregated to anatomical prostate regions and compared with systematic biopsy labels.

Recommended metrics:

- region-level AUC
- region-level AUPRC
- region-level balanced accuracy
- region-level sensitivity and specificity

#### Patient-level evaluation

Patient-level metrics evaluate whether the model can identify patients with clinically significant prostate cancer.

Recommended metrics:

- patient-level AUC
- patient-level AUPRC
- patient-level balanced accuracy
- sensitivity at fixed specificity

### 3.4 Main Comparisons

The experiments should be presented around the following comparisons.

#### Comparison 1: Single-supervision baselines

This comparison establishes the behaviour of each supervision source alone.

Expected interpretation:

- RA only is expected to provide stronger voxel-level localisation because of dense masks.
- TBx only may provide clinically meaningful target-level discrimination but weaker dense segmentation.
- SBx only may be limited spatially, but can provide useful patient- or region-level signal.

#### Comparison 2: Does adding biopsy supervision improve RA?

This comparison evaluates RA only versus RA + TBx, RA + SBx, and RA + TBx + SBx.

Main question:

> Does pathology-confirmed supervision improve a model trained with radiologist annotation?

This is important because RA provides lesion shape, while TBx and SBx provide histopathological confirmation.

#### Comparison 3: Are TBx and SBx complementary?

This comparison evaluates TBx only, SBx only, and TBx + SBx.

Main question:

> Can local targeted biopsy supervision and regional systematic biopsy supervision complement each other?

TBx is spatially more localised but incomplete. SBx is spatially coarse but samples broader anatomical regions. Combining them may provide a better pathology-guided lesion risk map.

#### Comparison 4: Full mixed supervision

The final comparison evaluates whether RA + TBx + SBx achieves the best overall performance.

Main question:

> Does combining voxel-level, target-ROI-level, and region-level supervision improve lesion risk prediction compared with any single supervision source?

### 3.5 Suggested Result Tables

#### Table 1. Supervision settings

| Model | RA | TBx | SBx | Supervision granularity |
|---|---|---|---|---|
| RA only | yes | no | no | voxel-level |
| TBx only | no | yes | no | target-ROI-level |
| SBx only | no | no | yes | region-level |
| RA + TBx | yes | yes | no | voxel + target ROI |
| RA + SBx | yes | no | yes | voxel + region |
| TBx + SBx | no | yes | yes | target ROI + region |
| RA + TBx + SBx | yes | yes | yes | voxel + target ROI + region |

#### Table 2. Lesion localisation performance

| Model | Dice | Sensitivity | Specificity | F1 |
|---|---:|---:|---:|---:|
| RA only |  |  |  |  |
| TBx only |  |  |  |  |
| SBx only |  |  |  |  |
| RA + TBx |  |  |  |  |
| RA + SBx |  |  |  |  |
| TBx + SBx |  |  |  |  |
| RA + TBx + SBx |  |  |  |  |

#### Table 3. Clinical-level performance

| Model | Patient AUC | Patient BAcc | Region AUC | Region BAcc |
|---|---:|---:|---:|---:|
| RA only |  |  |  |  |
| TBx only |  |  |  |  |
| SBx only |  |  |  |  |
| RA + TBx |  |  |  |  |
| RA + SBx |  |  |  |  |
| TBx + SBx |  |  |  |  |
| RA + TBx + SBx |  |  |  |  |

### 3.6 Expected Writing Logic

The experiment section should not simply report which model has the highest score. It should explain what each supervision source contributes.

A possible writing order is:

1. Establish the performance of single-supervision baselines.
2. Compare mixed-supervision models against single-supervision models.
3. Analyse whether RA improves localisation, whether TBx improves pathology-guided target discrimination, and whether SBx improves region-level prediction.
4. Discuss whether full mixed supervision provides the best trade-off across lesion-level, region-level, and patient-level metrics.

### 3.7 Main Claim to Test

The main claim of the paper can be written as:

> Mixed supervision from voxel-level RA, target-ROI-level TBx, and region-level SBx provides complementary information for prostate cancer lesion risk prediction on mpMRI.

The experimental section should support or qualify this claim by showing:

1. whether mixed supervision improves lesion localisation;
2. whether biopsy-confirmed labels improve clinical relevance;
3. whether region-level labels improve patient or region prediction;
4. whether the full mixed-supervision model performs better than single-source baselines.

