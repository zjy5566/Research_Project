import yaml,os,tqdm
from utils.zone_utils import *
from config import nii_dir
import nibabel as nib

def generate_localised_zones(zone_config_name, localised_level, nii_dir):
    patient_list = os.listdir(nii_dir)
    # cnt = 0
    for pid in tqdm.tqdm(sorted(patient_list)):
        # print(f"Processing patient: {pid}")
        t2_dir = os.path.join(nii_dir, pid, 't2.nii.gz')
        if not os.path.exists(t2_dir):
            print(f"Skipping {pid} as T2 image does not exist.")
            continue
        t2_img = nib.load(t2_dir)
        t2_arr = t2_img.get_fdata()

        gland_mask_dir = os.path.join(nii_dir, pid, f'gland.nii.gz')
        if not os.path.exists(gland_mask_dir):
            print(f"Skipping {pid} as gland mask does not exist.")
            continue
        gland_mask_arr = nib.load(gland_mask_dir).get_fdata()

        with open('zone_config.yml', 'r') as f:
            zone_config = yaml.safe_load(f)

        # get gland bbox limits
        x_min, x_max, y_min, y_max, z_min, z_max = bbox_range(t2_arr, gland_mask_arr)
        # get zone limits based on the localised level
        if localised_level == 20:
            zone_lim = config2coor_barzell(zone_config[zone_config_name], x_min, x_max, y_min, y_max, z_min, z_max)
        elif localised_level == 8:
            zone_lim = config2coor_8level(zone_config[zone_config_name], x_min, x_max, y_min, y_max, z_min, z_max)
        elif localised_level == 4:
            zone_lim = config2coor_4level(zone_config[zone_config_name], x_min, x_max, y_min, y_max, z_min, z_max)
        elif localised_level == 2:
            zone_lim = config2coor_2level(zone_config[zone_config_name], x_min, x_max, y_min, y_max, z_min, z_max)
        else:
            raise ValueError(f"Unsupported localised level: {localised_level}")
        zone_mask = gen_zone_on_mask(gland_mask_arr, zone_lim, z_min, z_max)
        zone_mask = zone_mask.astype('uint8')
        zone_mask_nii = nib.Nifti1Image(zone_mask, t2_img.affine, t2_img.header)
        nib.save(zone_mask_nii, os.path.join(nii_dir, pid, f'gland_zone_{localised_level}level_{zone_config_name}.nii.gz'))
        # cnt += 1
        # if cnt == 2:
        #     break

if __name__ == "__main__":
    zone_config_name = 'set1'  # zone configuration name, choose from 'set1', 'set2', or define your own
    # localised_level = 2  # localised level (20 for Barzell zones)
    for localised_level in [20, 8, 4, 2]:
        print(f"Generating localised zones for level {localised_level} with configuration {zone_config_name}")
        generate_localised_zones(zone_config_name, localised_level, nii_dir)