#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Preprocessing for membrane segmentation. Useful for, but not limited to 
preparation for Morse density tracing.

Overall it does the following:
  - Starting from bin 2 tomo, generates bin 4 and bin 8
  - Extracts a subtomogram that contains a region of interest, starting
  from coordinates specified the bin 2, 4, or 8 tomo
  - Extracts equivalent subtomograms from the other two tomos 

More details are given below:

The subtomo of interest can be defined by the following methods:

1) Rotation and crop method. The tomo is first rotated using an external
software and then in the rotated tomo the coordinates that define a 
box (crop) that contains the region of interest are determined. Meant
for use together with Matlab GUI tool for tomo rotation.

In this case, the subtomogram will contain the entire box of the rotated 
tomogram. If this box is slanted in the original (non-rotated) tomogram,
the subtomogram will be larger than the box. The mask showing the 
(possibly slanted) box is also generated, as well as the masked subtomogram.

Parameters required:
  - rotation, specified by Euler angles
  - box coordinates in the rotated tomogram

2) Slanted rectangle method. Coordinates of the (possibly slanted) box 
are determined directly from the original tomogram, that is without rotation.

To be precise, the box does not have to have right angles, so it can be
a 3D parallelogram.

The subtomogram is defined so that it contains the entire box. The mask
and the masked subtomos are generated like in the previous case.

Parameters required:
  - origin: coordinates of one corner of the box
  - basis: coordinates of all box corners (3 for 3D) that are directly 
  connected to the origin. Note that these do not need to be set so that 
  they define a proper (3D) rectangle, in general these will define a  
  (3D) parallelogram

3) Subtomo box specified method. Coordinates of the box (crop) that 
defines the subtomogram.

Here, no rotation is involved, and a mask and masked subtomograms are 
not generated.

Parameter required: 
  - box coordinates 

In all cases a parameter is specified that shows if the box coordinates
are given in the system where cooridnates start with 1 (as in many 
image viewers) or with 0.

If the shape of the specified bin2 tomo is not divisible by 4, it is 
cut as little as needed to make it divisible by 4 and this "cut" bin2
tomo is binned to make bin4 and bin8 tomos.

Similarly, if the shape of the subtomo (obtained as explained above)
is not divisible by 2 (if coordinates are specified for bin4 tomo) or
by 4 (if coordinates are specified for bin2 tomo), it is made wider
to make it compatible with bin 8.

By default the extracted subtomograms are normalized so that the values
 are between 0 amd 1 (mean 0.5, std 0.5/3, limits 0 and 1), as required 
for Morse density tracing. 

$Id$
Author: Vladan Lucic 
"""

import os

import numpy as np
import scipy as sp

import pyto
from pyto.geometry.rigid_3d import Rigid3D
from pyto.geometry.parallelogram import Parallelogram


##############################################################
#
# Parameters that need to be changed
#

#
# Subtomo definition
#
# As descriped above, there are three methods to define the subtomo of
# interest. Only one should be used and variable(s) for the other methods
# should be set to None. In all cases parameters are set for bin8 tomo.
#
# 1) The tomo is first rotated and then cropped

# Euler angles for rotation, order (phi, theta, psi) in degrees
# uses active extrinsic zxz Euler angles convention
euler_deg = [0, 0, 25]
euler_deg = None  # in case method 1 is not used

# box coordinates on the rotated tomo [[x_min, x_max], [y_min, ... ]]
after_rot_crop = [[199, 263], [97, 148], [94, 132]]
after_rot_crop = None  # in case method 1 is not used

# 2) Corners of a (likely) slanted rectangle are given

# one corner
corner_origin = [144,  124,  94]
#corner_origin = None  # in case method 2 is not used

# coordinates of the corners directly linked to the origin (axis 0 corners,
# axis 1 coordinate axes)
corners_basis = [[202,  97,  94],
                 [166, 170,  94],
                 [144,  123, 132]]
#corners_basis = None  # in case method 2 is not used

# 3) Just crop

# crop coordinates on the original tomo [[x_min, x_max], [y_min, ... ]]
simple_crop = None  # in case method 3 is not used

# Common for all

# bin factor of the tomo where subtomo definition coordinates are picked
crop_coord_bin = 8

# If the image viewer used to determine the crop parameters starts numbering
# pixels (and z slices) from 1 set this variable to 1 (like Tom), if it
# starts from 0 set it to 0
offset = 1

#
# File io
#

# bin 2 tomo file path; it has to end with 'bin2.mrc'
tomo_path = '../imod_rec/syn_m13-dko_11_bin2.mrc'

# output directory
out_dir = '../crop'

# if True, bin 4 and bin 8 tomos are read if they already exist and generated
# if they don't exist
read_existing = False


##############################################################
#
# Parameters that should stay the same for an entire project 
#

#
# File io
#

# suffix added to cropped tomo file names
crop_suffix = '_crop'

# suffix added to mask file names
mask_suffix = '_mask'

# suffix added to masked tomo file names
masked_suffix = '_masked'

# write logfile instead of printing info on stdout
logfile = True

# flag indicating if the tomo is read using memmap (default True)
memmap = True

#
# Greyscale
#

# tomo greyscale mean and std; None means no adjustement
mean = 0.5
std = 0.5 / 3

 # tomo greyscale values below the specified minimum and those above maximum
# are set to the minimum and maximum, respectively
min_value = 0
max_value = 1

# Sigma for gaussian smoothing that's applied to the mask before
# the subtomo is masked
gauss_sigma = 2


####################################################################
#
# Use only for debugging
#

# write subtomos
write_subtomo = True


###########################################################
#
# Work
#

def main():
    """
    Complete preprocessing.
    """

    # logging
    script_path = os.path.abspath(__file__)
    script_base, script_ext = os.path.splitext(script_path)
    if logfile:
        log_path = script_base + '.log'
        log_fd = open(log_path, 'w')
    else:
        log_fd = None
    print("Executing script {}".format(script_path), file=log_fd)

    # parse tomo path
    in_dir, tomo_name = os.path.split(tomo_path)
    base, extension = os.path.splitext(tomo_name)
    base_split = base.split('bin')
    try:
        if len(base_split) != 2:
            raise ValueError()
        base_root = base_split[0]
        bin_factor = int(base_split[1])
    except ValueError:
        raise ValueError(
            "Tomo file name has to end with 'binN.ext' where N is an int "
            + "and ext is an extension. The current tomo name is "
            + "{}".format(tomo_name))
    tomo_name_format = '{}bin{}{}'

    # read bin 2 tomo
    tomo_io = pyto.io.ImageIO()
    tomo_io.readHeader(file=tomo_path)
    in_pixel = tomo_io.pixelsize
    print(f"Reading bin 2 tomo {tomo_path}", file=log_fd)
    tomo_bin2 = pyto.grey.Image.read(file=tomo_path, memmap=memmap)

    # make and write bin 4x and 8x tomos
    pixel_local = in_pixel
    prev_tomo = tomo_bin2
    tomos = []
    for bin_fact in [4, 8]:

        out_tomo_name = tomo_name_format.format(base_root, bin_fact, extension)
        out_path = os.path.join(out_dir, out_tomo_name)
        if read_existing and os.path.exists(out_path):
            print(f"Reading bin {bin_fact} tomo {out_path}", file=log_fd)
            curr_tomo = pyto.grey.Image.read(file=out_path, memmap=memmap)
        else:    
            print("Making bin {} tomo".format(bin_fact), file=log_fd)
            curr_tomo = prev_tomo.bin(remain='correct', update=False)
            if (np.array(prev_tomo.data.shape) 
                > 2 * np.array(curr_tomo.data.shape)).any():
                print("\tTomo was cut to allow exact binning", file=log_fd)
            pixel_local = 2 * pixel_local
            print(f"\tWriting tomo to {out_path}", file=log_fd)
            tomo_io.write(file=out_path, data=curr_tomo.data, pixel=pixel_local)
        tomos.append(curr_tomo)
        prev_tomo = curr_tomo
    tomo_bin4, tomo_bin8 = tomos
 
    # determine crop type
    rot_crop_flag = False
    corner_crop_flag = False
    simple_crop_flag = False
    if (euler_deg is not None) and (after_rot_crop is not None):
        rot_crop_flag = True
    if (corner_origin is not None) and (corners_basis is not None):
        corner_crop_flag = True
    if simple_crop is not None:
        simple_crop_flag = True
    if np.array(
            [rot_crop_flag, corner_crop_flag, simple_crop_flag]).sum() != 1:
        raise ValueError(
            "Parameters for exactly one subtomo definition have to be "
            + "specified.")

    # determine crop_bin and large_mask for bin given by crop_coord_bin
    if rot_crop_flag:
        print("Determining subtomo box by rotation and crop method",
              file=log_fd)
 
        # box on rotated tomo
        after_rot_crop_binx = np.asarray(after_rot_crop) - offset
        rot_crop = Parallelogram.from_bounding_box(box=after_rot_crop_binx)

        # get rotation-center adjusted inverse rotation
        center = np.asarray(tomo_bin8.data.shape) // 2
        q = Rigid3D.make_r_euler(angles=np.asarray(euler_deg)*np.pi/180)
        af = pyto.geometry.Affine(gl=q, d=0)
        tf_0 = af.resetCenter(center=center)
        tf_0_inv = tf_0.inverse()

        # parallelogram on original (non-rotated) tomo
        orig_origin = tf_0_inv.transform(rot_crop.origin, xy_axes='point_dim')
        orig_basis = tf_0_inv.transform(rot_crop.basis, xy_axes='point_dim')
        large_mask = Parallelogram(origin=orig_origin, basis=orig_basis)

        # get crop coords for tomo bin crop_coord_bin
        shape_ar = (8 // crop_coord_bin) * np.array(
            tomo_bin8.data.shape, dtype=int)
        crop_bin = large_mask.get_bounding_box(shape=shape_ar)

    elif corner_crop_flag:
        print("Determining subtomo box by slanted rectangle method",
              file=log_fd)

        # make parallelogram
        large_mask = Parallelogram(
            origin=np.asarray(corner_origin)-offset,
            basis=np.asarray(corners_basis)-offset)

        # get crop coords for tomo bin crop_coord_bin
        shape_ar = (8 // crop_coord_bin) * np.array(
            tomo_bin8.data.shape, dtype=int)
        crop_bin = large_mask.get_bounding_box(shape=shape_ar)

    elif simple_crop_flag:
        print("Subtomo box specified", file=log_fd)
        crop_bin = np.asarray(simple_crop) - offset

    # convert crop_bin (from crop_coord_bin) to bin 8
    print(f"\tCroping box: {crop_bin}", file=log_fd)
    crop_bin8_float = crop_bin * crop_coord_bin / 8
    crop_bin8 = np.zeros_like(crop_bin)
    crop_bin8[:,0] = np.floor(crop_bin8_float).astype(int)[:,0]
    crop_bin8[:,1] = np.ceil(crop_bin8_float).astype(int)[:,1]
    if (crop_bin != crop_bin8).any():
        print(f"\tCroping box bin8 adjusted to {crop_bin8}", file=log_fd)

    # crop, adjust greyscale and write for different binnings
    out_tomo_name_format = '{}bin{}{}{}'
    for bin_fact, tomo in zip(
            [2, 4, 8],
            [tomo_bin2, tomo_bin4, tomo_bin8]):

        # current sizes 
        crop = 8 * crop_bin8 // bin_fact
        pixel_local = in_pixel * bin_fact / 2

        # crop tomo
        print("Processing bin {} tomo:".format(bin_fact), file=log_fd)
        inset = [slice(crop_1d[0], crop_1d[1]) for crop_1d in crop]
        print("\tCroping to {}".format(inset), file=log_fd)
        tomo_crop_data = tomo.useInset(inset=inset, update=False)
        subtomo = pyto.grey.Image(data=tomo_crop_data)

        # adjust greyscale
        print(
            ("\tNormalizing to mean {} and std {:.3f}, min value {} and "
             + "max value {}").format(mean, std, min_value, max_value),
            file=log_fd)
        subtomo.normalize(
            mean=mean, std=std, min_limit=min_value, max_limit=max_value)

        # write subtomo
        out_tomo_name = out_tomo_name_format.format(
            base_root, bin_fact, crop_suffix, extension)
        out_path = os.path.join(out_dir, out_tomo_name)
        print("\tWriting subtomo {}".format(out_path), file=log_fd)
        if write_subtomo:
            tomo_io.write(file=out_path, data=subtomo.data, pixel=pixel_local)

        if rot_crop_flag or corner_crop_flag:

            # make mask parallelogram for current bin and crop
            mask_origin = large_mask.origin * crop_coord_bin / bin_fact
            mask_origin = mask_origin - crop[:, 0]
            mask_basis = large_mask.basis * crop_coord_bin / bin_fact
            mask_basis = mask_basis - np.expand_dims(crop[:, 0], axis=0)
            mask = Parallelogram.make(
                origin=mask_origin, basis=mask_basis, shape=tomo_crop_data.shape, 
                thick=0, surface_label=1, outside_label=0, inside_label=1, 
                dtype='int16')

            # write mask
            out_tomo_name = out_tomo_name_format.format(
                base_root, bin_fact, crop_suffix+mask_suffix, extension)
            out_path = os.path.join(out_dir, out_tomo_name)
            print("\tWriting mask {}".format(out_path), file=log_fd)
            tomo_io.write(file=out_path, data=mask.data, pixel=pixel_local)

            # smooth mask
            if gauss_sigma is not None:
                mask.data = sp.ndimage.gaussian_filter(
                    mask.data, gauss_sigma, output='float32')

            # write masked outside black
            masked_data = subtomo.data * mask.data
            suffix = crop_suffix + masked_suffix + '-black'
            out_tomo_name = out_tomo_name_format.format(
                base_root, bin_fact, suffix, extension)
            out_path = os.path.join(out_dir, out_tomo_name)
            print(
                f"\tWriting black-masked subtomo {out_path}", file=log_fd)
            tomo_io.write(file=out_path, data=masked_data, pixel=pixel_local)

            # write masked outside white
            #masked_data = np.where(mask.data == 1, subtomo.data, 1)
            masked_data = masked_data + 1 - mask.data
            suffix = crop_suffix + masked_suffix + '-white'
            out_tomo_name = out_tomo_name_format.format(
                base_root, bin_fact, suffix, extension)
            out_path = os.path.join(out_dir, out_tomo_name)
            print(
                "\tWriting white-masked subtomo {out_path}", file=log_fd)
            tomo_io.write(file=out_path, data=masked_data, pixel=pixel_local)

    print("Finished", file=log_fd, flush=True)


# run if standalone
if __name__ == '__main__':
    main()
