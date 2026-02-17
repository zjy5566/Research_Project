import numpy as np
import os

root_dir = 'F:\\RP_dataset\\derived PROMIS data set'
nii_dir = os.path.join(root_dir,'MRI') # image directory
mri_report_dir = os.path.join(root_dir , 'PROMIS_OA_MRI_cleaned.xlsx') # MRI report directory
tpm_report_dir = os.path.join(root_dir, 'Template_biopsy') # TPM report directory

# Excluded Patient IDs
excluded_pids = ['P-14794814', 'P-81927032', 'P-50311284', 'P-53294571', 'P-31906541']

# Rules configuration
ccl_flag = 'uk'
rules_file = 'rules.yml'
total_cancer_defs = ['def1', 'def2', 'gs>=7']
# total_cancer_defs = ['def1', 'def2', 'gs>=7','gg4','ccl>=6','ccl>=4','gg>4','gg>=4+3','gg5']

# Confidence Interval Calculation
num_ci_iter = 100 # Number of bootstrap iterations for CI

# Sampling
sample_times = 100 

# Zone mappings from barzell zones to octant, quadrant, and hemi zones
# octant zone definitions
octant_zone = {
    1: [3, 9, 12],
    2: [1, 7, 11],
    3: [15, 19, 5, 12],
    4: [13, 17, 5, 11],
    5: [10, 4, 12],
    6: [2, 8, 11],
    7: [16, 20, 6, 12],
    8: [14, 18, 6, 11],
}
octant_half_zone = [5, 6]
octant_quater_zone = [11, 12]

# quadrant zone definitions 
quadrant_zone = {
    1: [3, 9, 12, 4, 10], # anterior right
    2: [1, 7, 11, 2, 8], # anterior left
    3: [15, 19, 5, 12, 20, 16, 6], # posterior right
    4: [14, 18, 6, 11, 13, 17, 5], # posterior left
}
quadrant_half_zone = [5, 6, 11, 12]

# 2-level zone definitions 
hemi_zone = {
    1: [3,5,9,12,15,19,4,6,10,16,20], # right
    2: [1,2,7,8,11,14,18,13,17,5,6], # left
        }
hemi_half_zone = [5,6]

# IoU thresholds
iou_thresholds = [np.finfo(float).tiny, 1e-5, 1e-3, 1e-2, 5e-2, 1e-1]
# iou_thresholds = [np.finfo(float).tiny,]

# PIRADS thresholds
pirads_thresholds = [2,3,4,5] 
# pirads_thresholds = [3]

current_zone_configs = ['set1',]
