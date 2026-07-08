# Method and Experiments Draft Framework

本框架基于当前项目代码、实验记录和结果文件整理，写作风格参考 CaPTion/MICCAI 类医学影像论文。当前报告主线建议从 **TCIA targeted-biopsy-confirmed lesion ROI supervision** 出发，而不是从传统 radiologist-only dense segmentation baseline 出发。

## 2. Method

### 2.1 Overview of the Proposed Framework

本研究提出一个基于 multiparametric MRI 的前列腺癌病灶风险图预测框架。模型输入为三通道 mpMRI，包括 T2-weighted MRI、diffusion-weighted imaging (DWI) 和 apparent diffusion coefficient (ADC)。模型输出为 voxel-level lesion risk map。与传统完全依赖 dense radiologist annotation 的分割方法不同，本研究重点探索如何利用 biopsy-confirmed supervision 训练病灶定位模型。

建议写作重点：

- 先强调任务目标：从 mpMRI 中预测 clinically significant prostate cancer (csPCa) lesion risk map。
- 再强调监督特点：TCIA targeted biopsy (TBx) 提供的是 biopsy-confirmed target ROI，而不是完整 dense lesion mask。
- 最后说明统一输出：所有监督源都优化同一个 lesion heatmap，避免多个输出头学习彼此割裂的任务。

可写成英文：

> We propose a 3D lesion-risk prediction framework for clinically significant prostate cancer localisation on multiparametric MRI. The input of the model consists of three MRI-derived channels, including T2-weighted imaging, diffusion-weighted imaging, and apparent diffusion coefficient maps. Instead of formulating the task as a purely dense segmentation problem, the proposed framework learns a voxel-level lesion risk map from heterogeneous supervision, with the main baseline built upon targeted-biopsy-confirmed lesion regions from the TCIA Prostate-MRI-US-Biopsy cohort.
>
> The key design of the framework is to use a single lesion output head. Dense lesion masks, targeted-biopsy-confirmed regions, and systematic-biopsy region labels, when available, are all used to supervise the same lesion heatmap. This design encourages the model to learn a unified representation of lesion likelihood rather than separate task-specific outputs.

### 2.2 Dataset and Label Sources

本报告建议把数据集分为三类监督源来写：

1. **TCIA targeted biopsy data**：当前 baseline 的核心数据。radiologist 首先标记 MRI suspicious lesion，随后通过 targeted biopsy 确认。target ROI 中的 label 由 biopsy-confirmed ISUP grade 决定。
2. **PUB radiologist dense lesion masks**：历史/参考监督源，用于 dense lesion segmentation baseline 或 legacy N-series comparison。
3. **Systematic biopsy region labels**：用于 weak regional supervision。通过 anatomical zones 与 systematic biopsy labels 对齐。
4. **PROMIS external validation**：作为外部验证或区域级评估数据源，注意需要区分 radiologist lesion-derived labels 与 biopsy-derived histopathology labels。

当前代码中的 label convention：

| Label | Meaning |
|---:|---|
| -1 | invalid / unsampled / no supervision |
| 0 | background / old negative placeholder |
| 1 | benign / negative biopsy |
| 2 | ISUP 1 |
| 3 | ISUP 2 |
| 4 | ISUP 3 |
| 5 | ISUP 4 |
| 6 | ISUP 5 |

当前 csPCa threshold 设为 `ISUP >= 2` 在原始医学意义上对应 label `>= 3`，代码中 `CSPC_THRESHOLD = 3`，因此报告中应写为：regions with ISUP grade group >= 2 were considered clinically significant prostate cancer.

可写成英文：

> The dataset integrates multiple prostate MRI cohorts with different annotation types. The main baseline uses the TCIA Prostate-MRI-US-Biopsy cohort, in which radiologist-defined target lesions are associated with targeted biopsy results. Each target ROI is assigned a biopsy-confirmed grade label. Voxels inside sampled target ROIs are treated as supervised voxels, while unsampled voxels outside the target ROIs are excluded from the targeted-biopsy loss.
>
> In addition to TCIA targeted biopsy supervision, radiologist-derived dense lesion masks and systematic-biopsy region labels are retained for comparison and ablation experiments. Dense masks provide voxel-level lesion supervision, whereas systematic biopsy labels provide region-level weak supervision through anatomical prostate zones.

### 2.3 Pre-processing

预处理部分建议写成一个完整数据管线：

1. 从 DICOM 中提取 T2 MRI、DWI 和 ADC。
2. 将多模态影像配准/整理为统一输入 tensor。
3. 由 prostate surface STL 或已有 mask 生成 gland mask。
4. 将 target biopsy metadata 或 target ROI 转换为 `target_mask.nii.gz`。
5. 将 systematic biopsy labels 整理为 12-zone 或 20-zone labels。
6. 将影像重采样至统一 spacing：`1.0 x 1.0 x 2.24 mm`。
7. 以 prostate gland 为中心裁剪为固定尺寸：`32 x 64 x 64`，输入 shape 为 `(3, 32, 64, 64)`。
8. 使用 gland foreground 进行 intensity normalisation。
9. 训练时使用 axial x-y plane flipping 作为数据增强，避免 z-axis flip 破坏 base-apex anatomy。

可写成英文：

> All MRI volumes were converted into a unified three-channel tensor consisting of T2-weighted, DWI, and ADC images. The volumes were resampled to a common spacing of 1.0 x 1.0 x 2.24 mm and cropped around the prostate gland to a fixed spatial size of 32 x 64 x 64 voxels. Intensity normalisation was performed within the gland foreground to reduce inter-case intensity variation. During training, random flipping along the axial in-plane directions was applied as data augmentation, while flipping along the superior-inferior axis was avoided to preserve prostate base-apex anatomy.
>
> For TCIA cases, targeted-biopsy-confirmed lesion regions were encoded in a target mask. Voxels inside sampled target ROIs were assigned labels according to the biopsy-confirmed grade, whereas voxels outside sampled target ROIs were treated as unsupervised for the TBx loss.

### 2.4 3D Residual U-Net Lesion-Risk Model

当前模型为 `ProstateSegMILNet`：

- backbone：3D residual U-Net。
- encoder：4 层 residual blocks + max pooling。
- bottleneck：residual block。
- decoder：transposed convolution upsampling + skip connections。
- normalisation：GroupNorm，适合小 batch 3D 医学影像训练。
- dropout：0.2。
- base channels：32。
- 输出头：一个 `1 x 1 x 1` convolution，输出 `lesion_logits`。

建议强调：single output head 是方法设计核心。旧版本中的 grade/gland 多头被移除，当前报告不要把多头网络作为主方法。

可写成英文：

> The lesion prediction model is based on a 3D residual U-Net architecture. The encoder consists of stacked 3D residual blocks followed by max-pooling operations, and the decoder reconstructs the spatial resolution using transposed convolutions and skip connections. Group normalisation is used instead of batch normalisation to improve training stability under small 3D batch sizes. The model contains a single learnable output head, implemented as a 1 x 1 x 1 convolution, which produces voxel-level lesion logits.
>
> This single-head design is central to the proposed framework. Rather than learning separate outputs for dense segmentation, biopsy classification, and regional prediction, all supervision sources are mapped to the same lesion-risk representation.

### 2.5 Region-Level MIL Pooling

当存在 systematic biopsy region labels 时，模型不会新增独立 region classification head，而是从 voxel-level lesion logits 中 pooling 出 region logits。

当前模型支持：

- log-mean-exp pooling (`MIL_POOLING = "lme"`)
- max pooling
- mean pooling

默认使用 log-mean-exp，`LME_R = 8.0`。对每个 anatomical zone，将该 zone 内的 voxel logits 聚合为一个 region-level logit，再与 systematic biopsy label 计算 loss。

可写成英文：

> To incorporate systematic-biopsy supervision, voxel-level lesion logits are aggregated into region-level predictions using multiple instance learning. For each anatomical prostate zone, the lesion logits inside the zone are pooled into a single region logit. The default pooling strategy is log-mean-exp pooling, which provides a differentiable approximation between mean and max pooling and allows high-risk voxels within a region to contribute more strongly to the regional prediction.
>
> This design enables region-level biopsy labels to supervise the voxel-level lesion heatmap without introducing an additional classification branch.

### 2.6 Loss Functions

当前 loss 为 `MixedSupervisionLoss`，包含四个 lesion-related branches：

| Loss branch | Supervision source | Current role |
|---|---|---|
| `lesion_dense` | PUB dense radiologist lesion mask | legacy/reference dense segmentation supervision |
| `lesion_sparse` | TCIA TBx-confirmed target ROI | main B1 baseline supervision |
| `lesion_sys` | SBx region labels | weak regional supervision / ablation |
| `lesion_outside_gland` | prostate gland mask | anatomical prior for suppressing extra-prostatic risk |

#### Dense lesion loss

用于 PUB dense lesion masks：

```text
L_dense = BCEWithLogits(lesion_logits, lesion_mask) + DiceLoss(lesion_logits, lesion_mask)
```

#### TCIA TBx sparse ROI loss

当前 B1 baseline 使用 sampled positive + sampled negative ROI BCE：

```text
valid(v) = target_mask(v) > 0
y(v) = 1[target_mask(v) >= CSPC_THRESHOLD]
L_TBx = mean_{v: valid(v)} BCEWithLogits(logit(v), y(v))
```

关键写法：未采样体素不是负样本，不参与 TBx loss。

#### SBx region MIL loss

对 systematic biopsy labels：

```text
region_logit(z) = MILPool({lesion_logit(v): v in zone z})
y(z) = 1[sys_label(z) >= CSPC_THRESHOLD]
L_SBx = BCE(region_logit(z), y(z)) + FocalLoss(region_logit(z), y(z))
```

#### Outside-gland anatomical prior

当前 B1 启用 outside-gland penalty，权重为 0.05：

```text
L_outside = mean_{v outside gland} softplus(logit(v))
```

该项只抑制腺体外不合理高风险，不对腺体内区域添加额外标签。

可写成英文：

> The total training objective is composed of lesion-related supervision terms. For dense radiologist masks, a combination of binary cross-entropy and Dice loss is used. For TCIA targeted-biopsy supervision, the loss is computed only over sampled target ROI voxels. Voxels with biopsy-confirmed csPCa labels are treated as positive, whereas sampled benign or below-threshold target voxels are treated as negative. Importantly, unsampled voxels outside target ROIs are excluded from the TBx loss and are not treated as background negatives.
>
> For systematic-biopsy supervision, voxel-level lesion logits are first pooled into anatomical-zone logits through MIL pooling. Region-level binary cross-entropy and focal loss are then applied to valid sampled zones. An additional outside-gland penalty is used in the TBx baseline to suppress high lesion risk outside the prostate gland.

### 2.7 Training Strategy

当前主要超参数：

| Item | Value |
|---|---:|
| Input channels | 3 |
| Input size | `(3, 32, 64, 64)` |
| Base channels | 32 |
| Dropout | 0.2 |
| Batch size | 4 |
| Epochs | 100 |
| Optimiser | Adam |
| Learning rate | 1e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Gradient clipping | 12.0 |
| Seed | 42 |
| Fixed specificity target | 0.95 |

可写成英文：

> All models were trained using the Adam optimiser with an initial learning rate of 1e-4 and weight decay of 1e-4. A cosine annealing learning-rate scheduler was applied over 100 epochs. The batch size was set to 4 due to the memory requirements of 3D MRI volumes. Gradient clipping with a maximum norm of 12.0 was used to stabilise training. The random seed was fixed to 42 for reproducibility.

## 3. Experiments

### 3.1 Experimental Aim

实验部分建议围绕一个核心问题写：

> Can biopsy-confirmed supervision train a clinically meaningful lesion risk map from mpMRI?

再拆成三个子问题：

1. TCIA TBx-confirmed target ROI supervision 能否作为合理 baseline？
2. 加入 SBx region labels 是否能改善 patient/region-level clinical metrics？
3. 与 PUB dense radiologist annotation 或 mixed supervision 相比，biopsy-confirmed supervision 的优势和限制是什么？

可写成英文：

> The experiments were designed to evaluate whether biopsy-confirmed supervision can train a clinically meaningful lesion risk map from mpMRI. We first established a TCIA targeted-biopsy baseline and then compared it with region-level systematic-biopsy supervision and legacy dense radiologist-mask supervision. The evaluation focuses on both voxel-level localisation and clinically relevant patient- and region-level classification metrics.

### 3.2 Experimental Settings

建议把实验分为 B-series 和 legacy N-series。

#### B-series: current main experiments

| Experiment | Training supervision | Purpose | Status |
|---|---|---|---|
| B1 | TCIA TBx-confirmed target ROI only | Main baseline | 已有主要结果 |
| B2 | TCIA SBx region labels only | Region-only weak supervision ablation | 待补/计划 |
| B3 | TCIA TBx target ROI + TCIA SBx regions | Test whether regional biopsy evidence improves localisation | 待补/计划 |

#### Legacy N-series: historical comparison

| Experiment | Training supervision | Purpose |
|---|---|---|
| N1 | PUB dense radiologist lesion masks | Dense segmentation reference |
| N2 | PUB dense masks + TCIA TBx | Dense + targeted biopsy comparison |
| N3 | PUB dense masks + TCIA SBx | Dense + regional weak supervision comparison |
| N4 | PUB dense masks + TCIA TBx + TCIA SBx | Mixed supervision comparison |

可写成英文：

> We organised the experiments into two groups. The B-series experiments represent the current biopsy-centred study design, where B1 uses TCIA targeted-biopsy-confirmed target ROIs as the main baseline, B2 uses systematic-biopsy region labels only, and B3 combines targeted-biopsy and systematic-biopsy supervision. The legacy N-series experiments are retained as reference comparisons to previous dense-mask-based settings.

### 3.3 Evaluation Metrics

建议按照层级写指标。

#### Voxel/lesion-level localisation

- Dice coefficient
- F1 score
- sensitivity
- specificity
- target csPCa Dice

注意：对于 B1，普通 lesion Dice 很低不一定代表 TBx ROI 判别完全失败，因为 B1 的监督不是完整 dense lesion mask。解释时要区分 dense segmentation quality 和 ROI discrimination ability。

#### TBx ROI-level discrimination

- TBx ROI balanced accuracy
- TBx ROI sensitivity / specificity
- TBx ROI AUC
- TBx ROI AUPRC
- sensitivity at fixed specificity = 0.95

这是当前 B1 最主要的正向结果载体。

#### Patient-level diagnosis

- patient balanced accuracy
- patient sensitivity / specificity
- patient AUC
- patient AUPRC
- sensitivity at fixed specificity
- specificity at fixed sensitivity

#### Region-level localisation / diagnosis

- region balanced accuracy
- region sensitivity / specificity
- region AUC
- region AUPRC

可写成英文：

> Model performance was evaluated at multiple levels. Voxel-level localisation was assessed using Dice coefficient, F1 score, sensitivity, and specificity when dense lesion masks were available. For the targeted-biopsy baseline, additional ROI-level metrics were reported, including balanced accuracy, AUC, AUPRC, and sensitivity at a fixed specificity of 0.95. Patient-level and region-level diagnostic performance were further evaluated using balanced accuracy, sensitivity, specificity, AUC, and AUPRC.

### 3.4 Result Presentation Plan

建议实验结果按以下顺序呈现，模仿参考论文的递进方式。

#### Table 1. Experimental settings

列出 B1/B2/B3/N1/N2/N3/N4 的监督源、loss branch、best model metric 和用途。

推荐列：

| Run | Dense mask | TBx ROI | SBx region | Outside gland prior | Best metric | Purpose |
|---|---|---|---|---|---|---|

#### Table 2. Main B1 quantitative results

列出 B1 在关键 epoch 的结果。当前已有数据可写：

| Epoch | Meaning | Val loss | Lesion Dice | Target csPCa Dice | TBx ROI AUC | TBx ROI AUPRC | TBx ROI sens@spec=0.95 | Patient AUC | Region AUC |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 28 | best val loss / lesion Dice | 0.630351 | 0.025162 | 0.012958 | 0.596069 | 0.689386 | 0.003915 | 0.502890 | 0.610035 |
| 67 | best target csPCa Dice | 0.752761 | 0.019394 | 0.015349 | 0.649923 | 0.775783 | 0.144917 | 0.467079 | 0.548195 |
| 80 | best TBx ROI AUC | 0.804729 | 0.017378 | 0.015326 | 0.664984 | 0.790246 | 0.165364 | 0.466563 | 0.539361 |
| 94 | best TBx ROI AUPRC | 待从 CSV 填入 | 待填 | 待填 | 待填 | 0.794167 | 待填 | 待填 | 待填 |
| 100 | final | 0.880279 | 0.014748 | 0.014621 | 0.659738 | 0.794018 | 0.172674 | 0.470846 | 0.546018 |

解读重点：

- TBx ROI AUC 从 0.531410 提升到 final 0.659738，最佳 0.664984。
- TBx ROI AUPRC 从 0.668189 提升到 final 0.794018，最佳 0.794167。
- TBx ROI sens@spec=0.95 从 0.023689 提升到 final 0.172674。
- patient/region 聚合表现仍弱，说明 ROI-level discrimination 尚未稳定传导到 clinical aggregation。

#### Table 3. Comparison with legacy N-series

可以放 N1/N2/N4 等实验的 best Dice、patient BAcc、region BAcc。当前已有 N2 和 N4 结果，N1 可从最新 result 补。

可重点报告：

- N2 best composite：`N2_CM05_FixedW005_Curr15_P09N005`，composite 0.4826，Dice 0.3454，Patient BAcc 0.7290，Region BAcc 0.5104。
- N4 best composite：`M05_EM10_Curr10_30_ClampN2P2_P085N010`，composite 0.4849，Dice 0.3398，Patient Sens 0.7833，Patient Spec 0.6095，Region Sens 0.1727，Region Spec 0.9549。
- N4 best Dice：`M02_FixedW010_025_Curr10_20_P09N005`，Dice 0.3431，Lesion Sens 0.4583，Lesion Spec 0.9917。

#### Fig. 1. Overall framework

建议画一个流程图：

```text
T2 / DWI / ADC
      |
      v
3D Residual U-Net
      |
      v
Voxel-level lesion logits
      |
      +--> Dense mask loss, if PUB mask available
      +--> TBx ROI BCE loss, if target ROI available
      +--> MIL pooling --> SBx region loss, if region labels available
      +--> Outside-gland penalty
```

#### Fig. 2. Training curves

展示：

- train/val loss
- target csPCa Dice
- TBx ROI AUC/AUPRC
- patient/region AUC

#### Fig. 3. Qualitative examples

展示输入 MRI、target ROI、gland mask、predicted lesion risk map。图注要说明不同监督源对应的可视化含义。

### 3.5 Draft Result Text for B1

这一段可以作为 B1 结果的初稿，后续你可以根据表格补数。

> The TCIA targeted-biopsy baseline was first evaluated to determine whether biopsy-confirmed target ROI supervision alone can train a lesion-risk map. The model showed limited dense segmentation performance when evaluated against radiologist-style lesion masks, with the best lesion Dice of 0.0252 at epoch 28 and the best target csPCa Dice of 0.0153 at epoch 67. This is expected to some extent because the B1 baseline is not trained with complete dense lesion masks; instead, supervision is restricted to sampled target ROI voxels.
>
> In contrast, the ROI-level discrimination metrics showed a clearer positive trend. The TBx ROI AUC increased from 0.5314 at the first epoch to 0.6597 at the final epoch, with a maximum value of 0.6650. Similarly, TBx ROI AUPRC increased from 0.6682 to 0.7940, reaching a maximum of 0.7942. Sensitivity at a fixed specificity of 0.95 also improved from 0.0237 to 0.1727. These results indicate that targeted-biopsy supervision provides useful discriminative signal within sampled target regions, even though it does not directly produce high-quality dense segmentation masks.
>
> However, the improvement at the TBx ROI level did not consistently translate to patient- or region-level diagnosis. Patient-level AUC remained close to chance level, and region-level AUC decreased after the early best epoch. This suggests that additional aggregation strategies, calibration, or region-level supervision may be required to convert voxel/ROI-level risk maps into reliable clinical-level predictions.

中文解释版：

> B1 的结果不要写成“分割失败”，而应写成“TBx ROI-level discrimination 有正向信号，但 dense segmentation 和 clinical aggregation 仍不足”。这样比较符合你的实验事实，也更像论文讨论方式。

### 3.6 Draft Comparison Text for N-series

> The legacy N-series experiments were used to contextualise the proposed biopsy-centred baseline against dense-mask and mixed-supervision settings. In the N2 experiments, combining PUB dense lesion masks with TCIA TBx supervision achieved the best composite score of 0.4826, with a lesion Dice of 0.3454 and patient-level balanced accuracy of 0.7290. In the N4 mixed-supervision experiments, the best composite score was 0.4849, obtained by the EM-weighted curriculum setting. The best lesion Dice in N4 was 0.3431, achieved by the fixed-weight curriculum setting.
>
> These results show that dense radiologist masks still provide substantially stronger voxel-level localisation supervision than TBx ROI supervision alone. Nevertheless, the B-series experiments remain clinically important because they evaluate whether biopsy-confirmed labels can serve as a more clinically grounded supervision source when dense radiologist masks are unavailable or inconsistent.

### 3.7 Suggested Experiment Section Structure

最终报告中 Experiments 部分建议这样安排：

#### 3.1 Experimental Setup

写数据划分、训练配置、模型选择指标。

#### 3.2 Evaluation Metrics

按 voxel / ROI / patient / region 四个层级写。

#### 3.3 TCIA TBx Baseline Results

主写 B1：

- loss 与 Dice 表现
- TBx ROI AUC/AUPRC
- fixed specificity 下 sensitivity
- patient/region 聚合问题

#### 3.4 Comparison with Mixed-Supervision Settings

写 N1/N2/N4 或 B2/B3：

- 如果 B2/B3 没跑完，就先写 legacy comparison。
- 如果 B2/B3 跑完，就把它们作为主 comparison，N-series 放 supplementary/reference。

#### 3.5 Qualitative Analysis

展示 risk map 可视化：

- 是否集中在 target ROI 附近
- 是否有 outside-gland false positive
- negative ROI 是否被压低
- patient/region aggregation 失败样本长什么样

#### 3.6 Discussion of Experimental Findings

建议讨论三个点：

1. **ROI discrimination improves**：说明 TBx labels 有效。
2. **Dense segmentation remains weak**：说明 sparse ROI supervision 不等于 dense mask supervision。
3. **Clinical aggregation needs further work**：说明 patient/region-level prediction 需要 pooling、calibration 或额外 loss。

### 3.8 Key Claims to Make Carefully

建议可以安全表达的 claim：

1. The TCIA TBx baseline provides a clinically grounded sparse supervision signal.
2. TBx ROI-level AUC and AUPRC improve during training.
3. Dense segmentation performance remains limited under TBx-only supervision.
4. Mixed/dense supervision achieves stronger voxel-level localisation metrics.
5. Patient- and region-level aggregation remains a limitation of the current model.

暂时不要强说的 claim：

1. 不要说 B1 已经实现 reliable lesion segmentation。
2. 不要说 patient-level diagnosis 已经有效，因为 patient BAcc 基本接近 0.5。
3. 不要说 B3 一定优于 B1，除非 B3 实验结果补齐。
4. 不要把 unsampled voxels 写成 confirmed negatives。

## 4. Immediate Writing Plan

下一步可以按这个顺序扩写：

1. 先完成 `Method 2.1-2.6`，因为这些已经由代码确定。
2. 再完成 `Experiments 3.1-3.3`，先写 B1 baseline 和评价指标。
3. 从 CSV 中补齐 B1 epoch 94、N1 latest 和必要的 comparison table。
4. 决定 B2/B3 是否作为正式结果。如果还没跑，就写为 future ablation，不放主结果表。
5. 最后写 Discussion，把当前结果解释成“TBx ROI 有判别信号，但 sparse supervision 到 dense/clinical-level prediction 之间仍有 gap”。

