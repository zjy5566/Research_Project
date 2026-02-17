import numpy as np

def bbox_range(arr, seg):
    assert arr.shape == seg.shape, "the shapes of img and seg are not equal"
    assert not (seg == 0).all(), "the values in the seg are all zeros"

    # Find the indices of all non-zero elements
    non_zero_indices = np.where(seg != 0)

    # Find the minimum and maximum indices along each axis
    x_min, x_max = np.min(non_zero_indices[0]), np.max(non_zero_indices[0])
    y_min, y_max = np.min(non_zero_indices[1]), np.max(non_zero_indices[1])
    z_min, z_max = np.min(non_zero_indices[2]), np.max(non_zero_indices[2])

    return x_min, x_max, y_min, y_max, z_min, z_max

def config2coor_barzell(config, x_min, x_max, y_min, y_max, z_min, z_max):
    anterior_split = [sum(config['anterior'][:i+1]) for i in range(len(config['anterior']))]
    posterior_split = [sum(config['posterior'][:i+1]) for i in range(len(config['posterior']))]

    z_cutoff = round(config['apex'] * (z_max - z_min) + z_min)
    y_cutoff = round(config['anterior-cutoff'] * (y_max - y_min) + y_min)
    left_cutoff = round(config['left-right'] * (x_max - x_min) + x_min)
    right_cutoff = round(x_max - config['left-right'] * (x_max - x_min))
    anterior_cutoff = [round(i*(right_cutoff - left_cutoff) + left_cutoff) for i in anterior_split]
    posterior_cutoff = [round(i*(right_cutoff - left_cutoff) + left_cutoff) for i in posterior_split]

    barzell_zone = {
        'apex': [1,3,5,7,9,13,15,17,19],
        'left': [11,],
        'right': [12,],
        'base': [2,4,6,8,10,14,16,18,20],
        'anterior': [9,10,3,4,1,2,7,8],
        'posterior': [19,20,15,16,5,6,13,14,17,18],
    }

    zone_lim = [{} for i in range(21)]
    for i in range(1,21):
        zone_lim[i]['z'] = (z_min, z_cutoff) if i in barzell_zone['apex'] else (z_cutoff+1, z_max) if i in barzell_zone['base'] else (z_min, z_max)
        zone_lim[i]['y'] = (y_min, y_cutoff) if i in barzell_zone['anterior'] else (y_cutoff, y_max) if i in barzell_zone['posterior'] else (y_min, y_max)

    zone_lim[12]['x'] = (x_min, left_cutoff)
    zone_lim[11]['x'] = (right_cutoff, x_max)

    anterior_zones = [(9,10),(3,4),(1,2),(7,8)]
    posterior_zones = [(19,20),(15,16),(5,6),(13,14),(17,18)]

    for i, zone_pair in enumerate(anterior_zones):
        # print(i, zone)
        for z in zone_pair:
            zone_lim[z]['x'] = (anterior_cutoff[i-1] if i > 0 else left_cutoff, anterior_cutoff[i])

    for i, zone_pair in enumerate(posterior_zones):
        for z in zone_pair:
            zone_lim[z]['x'] = (posterior_cutoff[i-1] if i > 0 else left_cutoff, posterior_cutoff[i])

    return zone_lim

def config2coor_8level(config, x_min, x_max, y_min, y_max, z_min, z_max):
    z_cutoff = round(config['apex'] * (z_max - z_min) + z_min)
    y_cutoff = round(config['anterior-cutoff'] * (y_max - y_min) + y_min)
    x_cutoff = round(0.5 * (x_max - x_min) + x_min)

    zone_lim = [{} for i in range(9)]
    eight_level_zone = {
        'apex': [1,2,3,4],
        'base': [5,6,7,8],
        'anterior':[1,2,5,6],
        'posterior':[3,4,7,8],
        'left':[2,4,6,8],
        'right':[1,3,5,7],
    }
    for i in range(1,9):
        zone_lim[i]['z'] = (z_min, z_cutoff) if i in eight_level_zone['apex'] else (z_cutoff+1, z_max) if i in eight_level_zone['base'] else (z_min, z_max)
        zone_lim[i]['y'] = (y_min, y_cutoff) if i in eight_level_zone['anterior'] else (y_cutoff, y_max) if i in eight_level_zone['posterior'] else (y_min, y_max)
        zone_lim[i]['x'] = (x_min, x_cutoff) if i in eight_level_zone['right'] else (x_cutoff+1, x_max) if i in eight_level_zone['right'] else (x_min, x_max)
    
    return zone_lim

def config2coor_4level(config, x_min, x_max, y_min, y_max, z_min, z_max):
    x_cutoff = round(0.5 * (x_max - x_min) + x_min)
    y_cutoff = round(config['anterior-cutoff'] * (y_max - y_min) + y_min)

    zone_lim = [{} for i in range(5)]
    four_level_zone = {
        'anterior': [1,2],
        'posterior': [3,4],
        'left': [2,4],
        'right': [1,3],
    }
    for i in range(1,5):
        zone_lim[i]['z'] = (z_min, z_max)
        zone_lim[i]['y'] = (y_min, y_cutoff) if i in four_level_zone['anterior'] else (y_cutoff, y_max) if i in four_level_zone['posterior'] else (y_min, y_max)
        zone_lim[i]['x'] = (x_min, x_cutoff) if i in four_level_zone['right'] else (x_cutoff+1, x_max) if i in four_level_zone['right'] else (x_min, x_max)

    return zone_lim

def config2coor_2level(config, x_min, x_max, y_min, y_max, z_min, z_max):
    x_cutoff = round(0.5 * (x_max - x_min) + x_min)

    zone_lim = [{} for i in range(3)]
    two_level_zone = {
        'right': [1],
        'left': [2],
    }
    for i in range(1,3):
        zone_lim[i]['z'] = (z_min, z_max)
        zone_lim[i]['y'] = (y_min, y_max)
        zone_lim[i]['x'] = (x_min, x_cutoff) if i in two_level_zone['right'] else (x_cutoff+1, x_max) if i in two_level_zone['left'] else (x_min, x_max)
    return zone_lim

def gen_zone_on_mask(gland_arr, zone_lim, z_min, z_max):
    zone_mask = np.zeros_like(gland_arr)
    for z in range(gland_arr.shape[2]): 
        if z in range(z_min, z_max + 1):
            # print(slice.shape, np.max(slice), np.min(slice))
            for x in range(0, zone_mask[:,:,z].shape[0]):
                for y in range(0, zone_mask[:,:,z].shape[1]):
                    if gland_arr[x, y, z] > 0.:
                        for i in range(1,21):
                            if (zone_lim[i]['x'][0] <= x <= zone_lim[i]['x'][1] and
                                zone_lim[i]['y'][0] <= y <= zone_lim[i]['y'][1] and 
                                zone_lim[i]['z'][0] <= z <= zone_lim[i]['z'][1]):
                                zone_mask[x, y, z] = i
                                break
    # print(np.max(zone_mask), np.min(zone_mask))
    return zone_mask
        