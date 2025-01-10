"""
Segmentation and analysis meta-data
"""


#######################################################
#
# Experiment description and file names
#

# identifier
identifier = 'cond_y_216'

# experimental condition
treatment = 'cond_y'

# batch and original identifier
batch = 'XY_2'

ori_identifier = 'tomo16_XY_2_y'

# synaptic vesicles
sv_file = '../../segmentation/XY_2/tomo16_XY_2_y/vesicles/tomo16_XY_2_y_vesicles.pkl'
sv_membrane_file = '../../segmentation/XY_2/tomo16_XY_2_y/vesicles/tomo16_XY_2_y_mem.pkl'
sv_lumen_file = '../../segmentation/XY_2/tomo16_XY_2_y/vesicles/tomo16_XY_2_y_lum.pkl'

# hierarchical segmentation of tethers and connectors
tethers_file = '../../segmentation/XY_2/tomo16_XY_2_y/connectors/tomo16_XY_2_y_new_AZ_good.pkl'
connectors_file = '../../segmentation/XY_2/tomo16_XY_2_y/connectors/tomo16_XY_2_y_new_rest_good.pkl'

# layers
layers_file =  '../../segmentation/XY_2/tomo16_XY_2_y/layers/balls-1_layers.dat'


########################################################
#
# Observations
#

# mitochondria in the presyn terminal
mitochondria = True


######################################################
#
# Microscopy
#

# microscope
microscope = 'Titan_2'

# pixel size
pixel_size = 1.756

# person who recorded the series
operator = 'someone'

# person who did membrane segmentation
segmented = 'someone'

#DDD or CCD
detector = 'DDD'
