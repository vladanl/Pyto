"""
Parameters and definitions needed for make_patterns and analyze_colocalizations

# Author: Vladan Lucic
"""

import numpy as np


#
# Parameters
#

# N particles
n_A = 300
n_B = 250
n_C = 50

# pixel size in nm
pixel_size_nm = 1

# geometry
region_x = [100, 500]
region_y = [200, 400]
region_z = 20
small_x = [200, 250]
small_y = [300, 350]
extend_x = [0, 1600]
extend_y = [100, 150]
extend2_x = [0, 4000]
extend2_y = [100, 120]
big_x = [120, 480]
big_y = [210, 390]
big_06_x = [150, 450]
big_06_y = [220, 380]
big_05_x = [175, 425]
big_05_y = [220, 380]

# regions
rectangle_50_area = 50 * 50
rectangle_50 = np.array([[small_x[0], small_y[0], region_z], 
                [small_x[1], small_y[1], region_z+1]])

rectangle_05_area = 250 * 160
rectangle_05 = np.array([[big_05_x[0], big_05_y[0], region_z], 
                  [big_05_x[1], big_05_y[1], region_z+1]])

rectangle_05w_05e_over = 140 * 160
rectangle_05sw_05ne_over = 140 * 140

rectangle_06_area = 300 * 160
rectangle_06 = np.array([[big_06_x[0], big_06_y[0], region_z], 
                  [big_06_x[1], big_06_y[1], region_z+1]])

rectangle_06w_06e_over = 240 * 160

rectangle_06sw_06ne_over = 240 * 140
rectangle_08_area = 360 * 180
rectangle_08 = np.array([[big_x[0], big_y[0], region_z], 
                  [big_x[1], big_y[1], region_z+1]])

rectangle_full_area = 400 * 200
rectangle_full = np.array([[region_x[0], region_y[0], region_z], 
                  [region_x[1], region_y[1], region_z+1]])


#
# Functions
#

def get_n_points_in_box(
        mps, pattern, tomo_id, box, domain_volume):
    """Gets N points of a pattern located in a smaller rectangular region.

    Arguments:
      - mps: (MultiParticleSets) particles
      - pattern: name of the point pattern
      - tomo_id: tomo_id
      - box: (ndarray shape 2, 3) box coordinates, specified as x, y, z
      coordinates of lower left and right up corners 
      - domain_volume: volume of the entire domain over which pattern
      is distributed

    Returns: actual N points in the box, expected N points in the box
    """

    # get all points
    pat_coords = (
        mps.particles.query(
            f"{mps.tomo_id_col} == @tomo_id "
            + f"and {mps.class_name_col} == @pattern")
        [mps.coord_reg_frame_cols]
        .to_numpy())
   
    # get n points in small
    box = np.asarray(box)
    xyz_min = box[0, :]
    xyz_max = box[1, :]
    box_shape = xyz_max - xyz_min
    cond = (
        (pat_coords[:, 0] >= xyz_min[0]) & (pat_coords[:, 0] < xyz_max[0])
        & (pat_coords[:, 1] >= xyz_min[1]) & (pat_coords[:, 1] < xyz_max[1])
        & (pat_coords[:, 2] >= xyz_min[2]) & (pat_coords[:, 2] < xyz_max[2]))
    actual = pat_coords[cond].shape[0]

    N_total = pat_coords.shape[0]
    box_volume = np.multiply.reduce(box_shape)
    expect = N_total * box_volume / domain_volume
    
    return actual, expect

def get_n_particles_in_small(
        pattern, N, small_x, small_y, big_x, big_y):
    """Gets N points of a pattern located in a smaller rectangular region.

    Depreciated.

    Used only if the pattern domain is rectangular.
    
    Arguments:
      - pattern: name of the point pattern
      - N: N points of the pattern in the larger region (big_x, big_y)
      - small_x, small_y: ([x_min, x_max], [y_min, y_max]) small
      rectangle coords
      - big_x, big_y: ([x_min, x_max], [y_min, y_max]) coords of
      the entire pattern domain (has to be a rectangle)
    """

    # get n points in small
    cond = (
        (pattern[:, 0] >= small_x[0]) & (pattern[:, 0] < small_x[1])
        & (pattern[:, 1] >= small_y[0]) & (pattern[:, 1] < small_y[1]))
    actual = pattern[cond].shape[0]

    expect = N * np.subtract(*small_x) * np.subtract(*small_y) / (
        np.subtract(*big_x) * np.subtract(*big_y))
    
    return actual, expect
