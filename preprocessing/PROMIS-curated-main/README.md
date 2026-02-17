
# Localization of clinically significant prostate cancer on multiparametric MR: an open reproducible analysis on the digitalized PROMIS dataset

Welcome to the repository for the fully digitalised open-source PROMIS study, along with the preprocessing tools and code for reproducing the quantitative analysis for two studies:
- A patient-level diagnostic accuracy described in the original paper [[Ahmed et al 2017]](https://doi.org/10.1016/S0140-6736(16)32401-1), Diagnostic accuracy of multi-parametric MRI and TRUS biopsy in prostate cancer (PROMIS): a paired validating confirmatory study, and
- A follow-up analysis on quantifying zone-level and lesion-level localization accuracy: [[anonymous]](https://currently-under-review) (under review; will be updated upon publication), Localization of clinically significant prostate cancer on multiparametric MR: an open reproducible analysis on the digitalized PROMIS dataset.

## What's included
- **PROMIS dataset**: An open-source fully digitalised dataset curated from the PROMIS study, including aligned radiological and histopathological labels.
- **Preprocessing tools**: Tools to prepare the dataset for subsequent automated quantitative localisation analysis and potentially machine learning tasks. 
- **Diagnostic accuracy analysis**: Code to reproduce the main results reported in the above two studies with their statistical analysis.

## The PROMIS dataset 
### Overview
| Item | Description |
| ---- | ----------- |
| Image Data Modality | T2-weighted, High-b DWI, ADC |
| Image Data Format | NifTi |
| Image Annotations | Lesion countours, prostate gland mask |
| Clinical report | Template biopsy report, radiologist readings |

### Download
You can download the dataset [here](https://zenodo.org/records/15683922).

## Generating local zones on the prostate masks
To generate local zones of different granularity, including hemi, quadrant, octant and Bazell zones, run the following script:

  ```bash
  python gen_localised_zones.py
  ```

## Diagnostic accuracy at patient-level and zone-levels
To compute the main analysis results:

1. **Specify configuration**  
   Define all required variables and directory paths in the `config.py` file. This includes paths to the dataset, output directories, and any relevant parameters.

2. **Run analysis**  
   Execute the main analysis script:

   ```bash
   python localised_analysis.py
   ```
