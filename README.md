# Research_Project
# 🚀 Prostate MRI Mixed-Supervision Preprocessing Pipeline

This repository contains the data preprocessing pipeline for Prostate Cancer detection and ISUP grading using Multi-parametric MRI (mpMRI). 

The pipeline is designed to fuse heterogeneous datasets (Targeted Biopsy, Systematic Biopsy, and Dense Segmentation) into a unified, **Mixed-Supervision** framework, inspired by *Rajagopal et al., 2024 ("Mixed Supervision of Histopathology Improves Prostate Cancer Classification From MRI")*.

---

## 📊 1. Overview of Datasets

Our pipeline unifies three distinct data sources into a standardized patient-centric format:

1. **TCIA Target Biopsy (Strong + Weak Supervision)**
   - **Targeted Cores**: Reconstructed as 3D needle trajectories with ISUP grades.
   - **Systematic Cores**: Mapped to a 12-zone anatomical mask (Apex/Mid/Base × Left/Right × Medial/Lateral).
2. **PROMIS Dataset (Weak Supervision)**
   - Systematic biopsy results mapped to a 20-zone prostate mask.
3. **MRI Dataset (Auxiliary Strong Supervision)**
   - Voxel-level Prostate Gland masks and Lesion segmentation masks.

---

## ⚙️ 2. Core Processing Pipeline

### Step 1: Modality Extraction & Registration
- **Smart Sequence Recognition**: Automatically parses complex and messy DICOM folders to identify `Axial T2`, `DWI` (High B-value), and `ADC` maps using robust keyword matching.
- **Co-Registration**: Uses SimpleITK's `MattesMutualInformation` to rigidly register ADC and DWI modalities to the high-resolution T2 space.

### Step 2: Spatial Normalization & Centroid-based Cropping
- **Resampling**: All modalities and masks are isotropically resampled to a uniform physical spacing of `[1.0, 1.0, 2.24] mm`.
- **Centroid Cropping**: Uses `sitk.LabelShapeStatisticsImageFilter` to extract the physical centroid of the prostate gland.
- **Dynamic Padding**: Crops a fixed volume of **$64 \times 64 \times 32$** around the centroid. Automatically applies zero-padding if the physical boundaries exceed the image size.
- **Normalization**: Applies Global Z-score normalization per channel `(x - mean) / std`.

### Step 3: Biopsy Trajectory Mapping (The Core Innovation)
Instead of manual pixel offsets, we utilize physical coordinate transformations (`TransformPhysicalPointToIndex`) to ensure perfect alignment:
- **Needle Masking**: Reconstructs 3D cylindrical needle tracks (1.5mm radius) using the Base and Tip coordinates from clinical CSV files.
- **Pathology Assignment**: Gleason scores are converted to ISUP Grade Groups (1-5). Intersecting needle tracks retain the maximum malignant grade.
- **Gland Constraint**: The needle masks are multiplied by the prostate `gland_mask` to rigidly filter out redundant trajectories that fall outside the prostate capsule.

### Step 4: Anatomical Zoning for MIL (Multiple Instance Learning)
- Automatically divides the prostate volume into 12 standardized regions based on physical LPS coordinates (Left/Right, Medial/Lateral, Apex/Mid/Base).
- Compiles systematic biopsy results into 1D numpy arrays (`systematic_labels.npy`) matching the generated zone masks.

---

## 📂 3. Unified Dataset Architecture

After running the pipeline, all data is standardized into the following structure, making it highly efficient for PyTorch `DataLoader`:

```text
Unified_Dataset/
├── TCIA_0001/
│   ├── input_tensor.npy         # (3, 32, 64, 64) -> [T2, DWI, ADC]
│   ├── target_bx.nii.gz         # Voxel-level target needle mask (ISUP labeled)
│   ├── zones_mask.nii.gz        # 1-12 anatomical zone mask
│   └── systematic_labels_12.npy # Array of shape (12,) with ISUP scores
│
├── PROMIS_P-10104751/
│   ├── input_tensor.npy
│   └── systematic_labels_20.npy # Array of shape (20,) with ISUP scores
│
├── PUB_000/
│   ├── input_tensor.npy
│   └── gland_mask.npy           # Voxel-level auxiliary segmentation mask
│
└── splits/
    ├── dataset_registry.csv     # Master lookup table
    ├── train.csv                # 70% Stratified Split
    ├── val.csv                  # 10% Stratified Split
    └── test.csv                 # 20% Stratified Split
