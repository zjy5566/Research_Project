from config import *
import pandas as pd
import nibabel as nib
import yaml,os

def get_res_from_rules(rules, data_dict):
    """get diagnostic result from different definitions"""
    res = -99
    for rule in rules:
        if eval(rule['criteria'], data_dict):
            res = rule['result']
            break
    return res


def load_patient_ids():
    """Load and filter patient IDs from the image directory."""
    patient_ids = [pid for pid in os.listdir(nii_dir) if pid.startswith('P-')]
    patient_ids = [pid for pid in patient_ids if pid not in excluded_pids]
    return patient_ids


def load_mri_report(patient_ids):
    """Load and process MRI report data."""
    mri_df = pd.read_excel(mri_report_dir)

    # get cancer significance from MRI report
    mri_dict = {}
    for pid in patient_ids:
        mri_data = mri_df[mri_df['patientID'] == pid]
        if mri_data.empty:
            print(f"Error: {pid} not found in MRI report.")
            continue
        else:
            sig = [i for i in mri_data['les_all'].values if not pd.isna(i)]
        mri_dict[pid] = sig
    return mri_dict


def load_tpm_data(pid):
    """Load template biopsy data for a given patientID."""
    tpm = None
    tpm_report = os.path.join(tpm_report_dir, f'{pid.upper()}.csv')

    if os.path.exists(tpm_report):
        tpm = pd.read_csv(tpm_report)
    if tpm is None:
        print(f"Cannot find CSV for patient {pid}")
    return tpm


def get_tpm_cancer_info(tpm, cancer_def_rules):
    """Extract cancer information from template biopsy based on rules."""
    zone_wc_values = tpm[tpm['zprescancer'] == 1]['zone_id']
    t_zone_wc = [int(v) for v in zone_wc_values.values if not pd.isna(v)]

    t_cancer_info = {}
    for zone in t_zone_wc:
        cancer_dict = {
            'gg1': tpm[tpm['zone_id'] == zone]['zprimgleason'].values[0] if not pd.isna(tpm[tpm['zone_id'] == zone]['zprimgleason'].values[0]) else -99,
            'gg2': tpm[tpm['zone_id'] == zone]['zsecondgleason'].values[0] if not pd.isna(tpm[tpm['zone_id'] == zone]['zsecondgleason'].values[0]) else -99,
            'ccl': tpm[tpm['zone_id'] == zone][f'maxcc{ccl_flag}'].values[0] if not pd.isna(tpm[tpm['zone_id'] == zone][f'maxcc{ccl_flag}'].values[0]) else -99,
        }
        t_cancer_info[zone] = get_res_from_rules(cancer_def_rules, cancer_dict)
    return t_zone_wc, t_cancer_info


def load_localised_mask(pid, zone_config, localised_level=20):
    """Load zone masks for a given patient and localised level."""
    file_path = os.path.join(nii_dir, pid, f'gland_zone_{localised_level}level_{zone_config}.nii.gz')
    if os.path.exists(file_path):
        return nib.load(file_path).get_fdata()
    return None


def load_mri_lesion_mask(pid):
    """Load MRI lesion mask. Default using a1 mask."""
    file_path = os.path.join(nii_dir, pid, 'lesion_a1.nii.gz')
    if os.path.exists(file_path):
        return nib.load(file_path).get_fdata()
    return None


def load_rules():
    """Loads rules from the rules.yml file."""
    return yaml.safe_load(open(rules_file))


def calculate_iou_3d(lesion_mask, zone_mask, iou_thre=0.05):
    """
    Calculate the IoU of each zone with the binary mask in 3D.
    
    Args:
        binary_mask (np.ndarray): A 3D binary mask with the same shape as zone_mask.
        zone_mask (np.ndarray): A 3D mask with zones labeled from 1 to 20.
        iou_thre (float): IoU threshold to consider a zone as relevant.
    
    Returns:
    dict: Dictionary where keys are zone labels and values are their IoU with the binary mask.
    """
    # Ensure masks have the same shape
    assert lesion_mask.shape == zone_mask.shape, "Masks must have the same shape"
   
    num_les = int(np.max(lesion_mask))

    iou_dict = {}
    for les in range(1, num_les+1):
        iou_dict[les] = []
        for zone in range(1, 21):
            binary_mask = (lesion_mask == les)
            zone_mask_current = (zone_mask == zone)
            
            # Calculate intersection and union
            intersection = np.logical_and(binary_mask, zone_mask_current).sum()
            union = np.logical_or(binary_mask, zone_mask_current).sum()
            
            if union != 0:
                iou = intersection / union
            else:
                iou = 0
            
            if iou > iou_thre:
                iou_dict[les].append(zone)
    
    return iou_dict


def calculate_cm(tpm_les_all, mri_les_all):
    """Calculates confusion matrix: True Positives, True Negatives, False Positives, False Negatives."""
    TP = np.sum((tpm_les_all == 1) & (mri_les_all == 1))
    TN = np.sum((tpm_les_all == 0) & (mri_les_all == 0))
    FP = np.sum((tpm_les_all == 0) & (mri_les_all == 1))
    FN = np.sum((tpm_les_all == 1) & (mri_les_all == 0))
    return TP, TN, FP, FN

def calculate_performance_metrics(TP, TN, FP, FN):
    """Calculates sensitivity, specificity, PPV, and NPV."""
    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    PPV = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    NPV = TN / (TN + FN) if (TN + FN) > 0 else np.nan
    return sensitivity, specificity, PPV, NPV

def bootstrap_confidence_intervals(mri_les_all, tpm_les_all, num_iterations=num_ci_iter):
    """Calculates bootstrap confidence intervals for performance metrics."""
    rng = np.random.default_rng(seed=42)
    sensitivities, specificities, PPVs, NPVs = [], [], [], []

    data_size = mri_les_all.shape[0]

    for _ in range(num_iterations):
        idx = rng.choice(data_size, data_size, replace=True)
        mri_les_all_sample = mri_les_all[idx]
        tpm_les_all_sample = tpm_les_all[idx]

        TP, TN, FP, FN = calculate_cm(tpm_les_all_sample, mri_les_all_sample)
        sensitivity, specificity, PPV, NPV = calculate_performance_metrics(TP, TN, FP, FN)

        sensitivities.append(sensitivity)
        specificities.append(specificity)
        PPVs.append(PPV)
        NPVs.append(NPV)
    
    # Remove NaNs in case of any invalid bootstrap samples (e.g., division by zero)
    sensitivities = np.array(sensitivities)[~np.isnan(sensitivities)]
    specificities = np.array(specificities)[~np.isnan(specificities)]
    PPVs = np.array(PPVs)[~np.isnan(PPVs)]
    NPVs = np.array(NPVs)[~np.isnan(NPVs)]

    # Calculate 95% confidence intervals (2.5th and 97.5th percentiles)
    sens_ci = np.percentile(sensitivities, [2.5, 97.5])
    spec_ci = np.percentile(specificities, [2.5, 97.5])
    PPV_ci = np.percentile(PPVs, [2.5, 97.5])
    NPV_ci = np.percentile(NPVs, [2.5, 97.5])

    return sens_ci, spec_ci, PPV_ci, NPV_ci


def format_log_dict(log_dict):
    """Formats the numerical values in the log dictionary to percentages and adds CI strings."""
    for key in log_dict:
        if key == 'iou_thre':
            continue
        if isinstance(log_dict[key], float):
            log_dict[key] = round(log_dict[key] * 100, 2)
        elif isinstance(log_dict[key], tuple):
            log_dict[key] = tuple(round(x * 100, 2) for x in log_dict[key])
        elif isinstance(log_dict[key], np.ndarray):
            log_dict[key] = tuple(round(x * 100, 2) for x in log_dict[key])
    
    log_dict['sensitivity_with_ci'] = f"{log_dict['sensitivity']} ({log_dict['sensitivity_ci'][0]}, {log_dict['sensitivity_ci'][1]})"
    log_dict['specificity_with_ci'] = f"{log_dict['specificity']} ({log_dict['specificity_ci'][0]}, {log_dict['specificity_ci'][1]})"
    log_dict['PPV_with_ci'] = f"{log_dict['PPV']} ({log_dict['PPV_ci'][0]}, {log_dict['PPV_ci'][1]})"
    log_dict['NPV_with_ci'] = f"{log_dict['NPV']} ({log_dict['NPV_ci'][0]}, {log_dict['NPV_ci'][1]})"
    return log_dict