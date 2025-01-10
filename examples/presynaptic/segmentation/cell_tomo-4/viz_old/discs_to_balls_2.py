#!/usr/bin/env python
"""
Reads a labels file containing discs and writes another labels file with the
discs extending to balls having same centers and radii as the discs.

This script may be placed anywhere in the directory tree.

ToDo:
  - see about excentricity

$Id$
Author: Vladan Lucic (Max Planck Institute for Biochemistry) 
"""
from __future__ import unicode_literals
#from builtins import str

__version__ = "$Revision$"

import os
import os.path
import time
import logging

import numpy
import scipy
import scipy.ndimage as ndimage

import pyto
import pyto.scripts.common as common

# import tomo_info
tomo_info = common.__import__(name='tomo_info', path='../common')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    datefmt='%d %b %Y %H:%M:%S')



##############################################################
#
# Parameters
#
##############################################################

############################################################
#
# Image (grayscale) file. Used only to read image size (shape). Should be in 
# em or mrc format.
#

# name of the image file
if tomo_info is not None: image_file_name = tomo_info.image_file_name
#image_file_name = "../3d/tomo.em"

#####################################################
#
# Labels file containing discs. If the file is in em or mrc format shape, data
# type, byte order and array order are not needed (should be set to None). If
# these variables are specified they will override the values given in em or
# mrc headers.

# name of the input (discs) file 
discs_file_name = "disks_2.raw"   

# name of the output (balls) file 
if tomo_info is not None: balls_file_name = tomo_info.labels_file_name
balls_file_name = "labels_2.mrc"    

# file dimensions
shape = None   # shape given in header of the disc file (if em
               # or mrc), or in the tomo (grayscale image) header   
#shape = (100, 120, 90) # shape given here

# discs file data type (e.g. 'int8', 'int16', 'int32', 'float16', 'float64') 
discs_data_type = 'uint8'
 
# balls file data type (e.g. 'int8', 'int16', 'int32', 'float16', 'float64')
# For mrc files use 'uint8' if less than 256 lables, or 'int16' for more labels
balls_data_type = 'uint8'
 
# file byte order ('<' for little-endian, '>' for big-endian)
byte_order = '<'

# file array order ('FORTRAN' for x-axis fastest, 'C' for z-axis fastest)
array_order = 'FORTRAN'

#########################################
#
# Calculation related parameters
#

# ids of all segments  
#if tomo_info is not None: all_ids = tomo_info.all_ids
all_ids =  [2,3] + [35, 47, 72, 74] + [36, 48, 49, 50, 52, 58, 59, 60, 61]
#all_ids = [2,3,6]       # individual ids
#all_ids = range(2,64)  # range of ids
#all_ids = None         # all segments are to be used

# ids of all discs (all formats given for all_ids can be used)
#if tomo_info is not None: vesicle_ids = tomo_info.vesicle_ids
vesicle_ids = [52, 58, 59, 60, 61]
#vesicle_ids = [3,6]

# id or the region surrounding discs
if tomo_info is not None: surround_id = tomo_info.segmentation_region
#surround_id = 2

# first covert balls to discs
balls_to_discs = False

# magnify disks by a factor (before making balls)
mag_factor = 1

# principal axis of the discs
disc_axis = -1    # along z-axis, discs in x-y plane

# normal vector of the plane that cuts the shpere
normal_vector = [0, 0, 1]  # along z-axis

# angle that denotes how much of the ball is cut by the planes (in radians)
theta = None            # the ball is not cut
#theta = numpy.pi/6   # 30 deg
#theta = numpy.pi/2   # 90 deg, a half ball is cut by each plane 

# increase radii of all discs by this ammount
enlarge_radii = 0.5

# check if all discs exist, have exactly one connected part and don't overlap
check_discs = True


################################################################
#
# Work
#
################################################################


################################################################
#
# Main function
#

def main():
    """
    Main function
    """

    # read image and vesicles
    image = common.read_image(file_name=image_file_name, memmap=True)
    new_shape = common.find_shape(file_name=image_file_name, shape=shape,
                       suggest_shape=image.data.shape)
    # read input file
    discs = pyto.segmentation.Ball.read(
        file=discs_file_name, ids=all_ids, byteOrder=byte_order, 
        dataType=discs_data_type, arrayOrder=array_order, shape=new_shape)

    # make discs if needed
    if balls_to_discs:
        discs.thinToMaxDiscs(ids=vesicle_ids, external=surround_id)
    
    # check
    if check_discs:
        topo = pyto.segmentation.Topology(segments=discs, ids=vesicle_ids)
        n_connected = topo.calculateHomologyRank(dim=0)[vesicle_ids]
        if not (n_connected == 1).all():
            many = topo.ids[numpy.nonzero(n_connected[topo.ids]>1)[0]]
            if len(many) > 0:
                logging.warning(
                    "The following discs are disconnected: " + str(many))
            zero = topo.ids[numpy.nonzero(n_connected[topo.ids]<1)[0]]
            if len(zero) > 0:
                logging.warning(
                    "The following discs do not exist: " + str(zero))

    # magnify if needed
    if mag_factor > 1: 
        discs.magnify(factor=mag_factor)

    # convert discs to balls
    balls = discs.extendDiscs(
        ids=vesicle_ids, axis=disc_axis, cut=normal_vector, theta=theta, 
        enlarge=enlarge_radii, external=surround_id, check=check_discs)

    # write balls file
    balls.write(file=balls_file_name, dataType=balls_data_type)


# run if standalone
if __name__ == '__main__':
    main()
