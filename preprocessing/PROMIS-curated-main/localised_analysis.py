# combined_zone_analysis.py
import pandas as pd
import numpy as np
import tqdm
import random
from utils.stat_utils import *
from config import *

# Helper function to calculate ratio of intersection for multi-level zones
def calculate_ratio(multi_zone_dict, big_zone_mask, bar_zone_mask, ratio_dict):
    """Calculates the ratio of intersection between combined zones and their corresponding barzell zones."""
    for bar_zone, big_zone in multi_zone_dict.items():
        for zone in big_zone:
            intersection = np.logical_and(big_zone_mask == zone, bar_zone_mask == bar_zone).sum()
            ratio = intersection / np.sum(bar_zone_mask == bar_zone)
            ratio_dict[bar_zone][zone] = ratio  
    return ratio_dict
    

def run_analysis_for_localised_level(localised_level: int):
    """
    Runs the analysis for a specified zone level (2, 4, 8, or 20).

    Args:
        localised_level (int): The zone level to perform analysis for (2, 4, 8, or 20).
    """
    print(f"\n--- Running Analysis for {localised_level}-Zone Level ---")
    random.seed(42) # Ensure reproducibility

    patient_ids = load_patient_ids()
    mri_dict = load_mri_report(patient_ids)
    rules = load_rules()

    # --- Zone Level Specific Configurations ---
    num_zones = 20
    tpm_zone_map_config = None # Holds mapping dictionaries for 2, 4 or 8 zone
    
    if localised_level == 20:
        num_zones = 20
        zone_level_filename_part = "barzell_zone_level"
        
    elif localised_level == 8:
        num_zones = 8
        zone_level_filename_part = "8_level"
        # Pre-calculate reverse and group mappings for 8-zone
        reverse_map = {v: k for k, values in octant_zone.items() for v in values}
        half_map = {zone: [] for zone in octant_half_zone}
        quarter_map = {zone: [] for zone in octant_quater_zone}
        for zone_id_20 in octant_half_zone:
            for k, v in octant_zone.items():
                if zone_id_20 in v:
                    half_map[zone_id_20].append(k)
        for zone_id_20 in octant_quater_zone:
            for k, v in octant_zone.items():
                if zone_id_20 in v:
                    quarter_map[zone_id_20].append(k)
        # Remove half and quarter zones from reverse_map as they are handled by their specific maps
        for zone_to_remove in octant_half_zone + octant_quater_zone:
            reverse_map.pop(zone_to_remove, None)
        tpm_zone_map_config = {
            "type": "eight_zone",
            "reverse_map": reverse_map,
            "half_map": half_map,
            "quarter_map": quarter_map,
        }
    elif localised_level == 4:
        num_zones = 4
        zone_level_filename_part = "4_level"
        # Pre-calculate reverse and group mappings for 4-zone
        reverse_map = {v: k for k, values in quadrant_zone.items() for v in values}
        half_map = {zone: [] for zone in quadrant_half_zone}
        for zone_id_20 in quadrant_half_zone:
            for k, v in quadrant_zone.items():
                if zone_id_20 in v:
                    half_map[zone_id_20].append(k)
        # Remove half zones from reverse_map
        for zone_to_remove in quadrant_half_zone:
            reverse_map.pop(zone_to_remove, None)
        tpm_zone_map_config = {
            "type": "four_zone",
            "reverse_map": reverse_map,
            "half_map": half_map,
        }
    elif localised_level == 2:
        num_zones = 2
        zone_level_filename_part = "2_level"
        # Pre-calculate reverse and group mappings for 2-zone
        reverse_map = {v: k for k, values in hemi_zone.items() for v in values}
        half_map = {zone: [] for zone in hemi_half_zone}
        for zone_id_20 in hemi_half_zone:
            for k, v in hemi_zone.items():
                if zone_id_20 in v:
                    half_map[zone_id_20].append(k)
        # Remove half zones from reverse_map
        for zone_to_remove in hemi_half_zone:
            reverse_map.pop(zone_to_remove, None)
        tpm_zone_map_config = {
            "type": "four_zone",
            "reverse_map": reverse_map,
            "half_map": half_map,
        }
    else:
        raise ValueError("Invalid localised_level. Must be 2, 4, 8, or 20.")

    for zone_config in current_zone_configs:
        log_dict_array_for_current_file = []

        for pirads_thre in pirads_thresholds:
            for iou_thre in iou_thresholds:
                print(f"Processing: Zone Config='{zone_config}', Pirads={pirads_thre}, IoU={iou_thre:.1e}")
                
                tpm_les_dict_all_patients = {k: [] for k in total_cancer_defs}
                mri_les_all_patients = []

                for pid in tqdm.tqdm(patient_ids, desc="    Patients"):
                    # Load zone masks (current level and 20 barzell zones for calculating ratios)
                    zone_arr_current_level = load_localised_mask(pid, zone_config, localised_level)
                    zone_arr_20 = load_localised_mask(pid, zone_config, ) # Always needed for ratio calcs
                    if zone_arr_current_level is None or zone_arr_20 is None:
                        print(f"    Skipping patient {pid}: Missing zone mask files.")
                        continue

                    # Process MRI Lesions
                    mri_les_arr = load_mri_lesion_mask(pid)
                    if mri_les_arr is None:
                        mri_les_arr = np.zeros_like(zone_arr_current_level)

                    mri_les_cnt = 0
                    for num,sig in enumerate(mri_dict[pid]):             
                        if sig >= pirads_thre:
                            mri_les_arr[mri_les_arr == num+1] = mri_les_cnt+1
                            mri_les_cnt += 1
                        else:
                            mri_les_arr[mri_les_arr == num+1] = 0
                    iou_results = calculate_iou_3d(mri_les_arr, zone_arr_current_level, iou_thre)
                    mri_les_level_specific = np.zeros(num_zones)
                    for _, zones_overlap in iou_results.items(): # iou_results maps lesion_label to list of overlapping zone_ids
                        for zone_id in zones_overlap:
                            mri_les_level_specific[zone_id - 1] = 1 # Mark zone as having an MRI lesion
                    
                    # Append MRI lesion status, applying sampling if relevant
                    mri_les_all_patients.append(mri_les_level_specific)
                    if localised_level in [2, 4, 8]:
                        for _ in range(sample_times - 1):
                            mri_les_all_patients.append(mri_les_level_specific)

                    # Process TPM(template mapped biopsy) Data
                    tpm = load_tpm_data(pid)
                    if tpm is None:
                        print(f"    Skipping patient {pid}: Missing TPM data.")
                        continue
                    
                    # Calculate ratios for 4/8 zone if applicable (used for probabilistic mapping)
                    half_ratio_dict = {}
                    quarter_ratio_dict = {}
                    if localised_level == 8:
                        half_ratio_dict = {k:{v:None for v in vs} for k,vs in tpm_zone_map_config["half_map"].items()}
                        half_ratio_dict = calculate_ratio(tpm_zone_map_config["half_map"], zone_arr_current_level, zone_arr_20, half_ratio_dict)
                        quarter_ratio_dict = {k:{v:None for v in vs} for k,vs in tpm_zone_map_config["quarter_map"].items()}
                        quarter_ratio_dict = calculate_ratio(tpm_zone_map_config["quarter_map"], zone_arr_current_level, zone_arr_20, quarter_ratio_dict)
                    elif localised_level == 4 or localised_level == 2:
                        half_ratio_dict = {k:{v:None for v in vs} for k,vs in tpm_zone_map_config["half_map"].items()}
                        half_ratio_dict = calculate_ratio(tpm_zone_map_config["half_map"], zone_arr_current_level, zone_arr_20, half_ratio_dict)

                    # Apply cancer definitions to TPM data
                    for cancer_def in total_cancer_defs:
                        cancer_def_rules = rules['cancer_definition'][cancer_def]
                        t_zone_wc_20_level, t_cancer_info_20_level = get_tpm_cancer_info(tpm, cancer_def_rules)
                        curr_zone_wc_20_level = [zone for zone in t_zone_wc_20_level if t_cancer_info_20_level[zone] == '1']
                        
                        if localised_level == 20: # Direct mapping for 20-zone
                            tpm_les_level_specific = np.zeros(num_zones)
                            for zone_id_20 in curr_zone_wc_20_level:
                                tpm_les_level_specific[zone_id_20 - 1] = 1
                            tpm_les_dict_all_patients[cancer_def].append(tpm_les_level_specific)
                        else: 
                            for _ in range(sample_times):
                                tpm_les_level_specific = np.zeros(num_zones)
                                for zone_id_20 in curr_zone_wc_20_level:
                                    if t_cancer_info_20_level[zone_id_20] == '1':
                                        new_zone_id_mapped = None
                                        if zone_id_20 in tpm_zone_map_config["half_map"]:
                                            new_zone_id_mapped = random.choice(tpm_zone_map_config["half_map"][zone_id_20])
                                        elif localised_level == 8 and zone_id_20 in tpm_zone_map_config["quarter_map"]: # for 8-zone quarter mapping only
                                            rand_num = random.random()
                                            q_zones = tpm_zone_map_config["quarter_map"][zone_id_20]
                                            ordered_ratios = np.array([quarter_ratio_dict[zone_id_20][qz] for qz in q_zones])
                                            cumulative_ratios = np.cumsum(ordered_ratios)
                                            selected_zone_index = np.searchsorted(cumulative_ratios, rand_num, side='right')
                                            if selected_zone_index == len(q_zones):
                                                selected_zone_index -= 1
                                            new_zone_id_mapped = q_zones[selected_zone_index]
                                            # if new_zone_id_mapped is None and q_zones: # Fallback for edge cases with float precision
                                            #     new_zone_id_mapped = q_zones[-1]
                                        else: # Direct mapping from 20-zone to 2/4/8 zone
                                            new_zone_id_mapped = tpm_zone_map_config["reverse_map"].get(zone_id_20)
                                            
                                        if new_zone_id_mapped is not None:
                                            tpm_les_level_specific[new_zone_id_mapped - 1] = 1
                                tpm_les_dict_all_patients[cancer_def].append(tpm_les_level_specific)
                mri_les_all_patients_np = np.array(mri_les_all_patients)

                for cancer_def in total_cancer_defs:
                    tpm_les_all_patients_np = np.array(tpm_les_dict_all_patients[cancer_def])
                    
                    # Check for shape consistency before calculation
                    if tpm_les_all_patients_np.shape != mri_les_all_patients_np.shape:
                        print(f"    Skipping {cancer_def} due to shape mismatch: TPM {tpm_les_all_patients_np.shape} vs MRI {mri_les_all_patients_np.shape}")
                        continue

                    TP, TN, FP, FN = calculate_cm(tpm_les_all_patients_np, mri_les_all_patients_np)
                    sensitivity, specificity, PPV, NPV = calculate_performance_metrics(TP, TN, FP, FN)

                    log_dict = {
                        "definition": cancer_def,
                        "pirads_thre": pirads_thre,
                        "iou_thre": iou_thre,
                        "TP": TP, "FP": FP, "FN": FN, "TN": TN,
                        "sensitivity": sensitivity, "specificity": specificity,
                        "PPV": PPV, "NPV": NPV,
                    }
                    
                    # Print raw counts and averages per zone
                    print(f"    {cancer_def} - Raw Counts (TP/TN/FP/FN): {TP}, {TN}, {FP}, {FN}")
                    print(f"    {cancer_def} - Avg per Zone (TP/TN/FP/FN): {TP/num_zones:.2f}, {TN/num_zones:.2f}, {FP/num_zones:.2f}, {FN/num_zones:.2f}")
                    print(f"    {cancer_def} - Metrics: Sensitivity: {sensitivity:.2f}, Specificity: {specificity:.2f}, PPV: {PPV:.2f}, NPV: {NPV:.2f}")

                    # Bootstrap confidence intervals
                    sens_ci, spec_ci, PPV_ci, NPV_ci = bootstrap_confidence_intervals(mri_les_all_patients_np, tpm_les_all_patients_np)
                    log_dict['sensitivity_ci'] = sens_ci
                    log_dict['specificity_ci'] = spec_ci
                    log_dict['PPV_ci'] = PPV_ci
                    log_dict['NPV_ci'] = NPV_ci
                
                    log_dict_array_for_current_file.append(format_log_dict(log_dict))
        
        # --- Save Results to Excel ---
        df = pd.DataFrame(log_dict_array_for_current_file)
        
        # Construct the filename similar to original scripts
        filename = f"{zone_level_filename_part}_0_{zone_config}_multiiou.xlsx"
        df.to_excel(filename, index=False)
        print(f"Saved results to {filename}")


if __name__ == "__main__":
    # Define which zone levels to run the analysis for
    localised_levels_to_run = [ 20, 8, 4, 2] 
    # localised_levels_to_run = [2, ]
    for level in localised_levels_to_run:
        run_analysis_for_localised_level(level)