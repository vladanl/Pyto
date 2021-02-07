"""
Segmentation and analysis meta-data
"""


#######################################################
#
# Experiment description and file names
#

# identifier
identifier = 'cond_y_301'

# experimental condition
treatment = 'cond_y'

# batch and original identifier
batch = 'XY_3'

ori_identifier = 'tomo1_XY_3_y'

# synaptic vesicles
sv_file = '../../segmentation/XY_3/tomo1_XY_3_y/vesicles/tomo1_XY_3_y_vesicles.pkl'
sv_membrane_file = '../../segmentation/XY_3/tomo1_XY_3_y/vesicles/tomo1_XY_3_y_mem.pkl'
sv_lumen_file = '../../segmentation/XY_3/tomo1_XY_3_y/vesicles/tomo1_XY_3_y_lum.pkl'

# hierarchical segmentation of tethers and connectors
tethers_file = '../../segmentation/XY_3/tomo1_XY_3_y/connectors/tomo1_XY_3_y_new_AZ_good.pkl'
connectors_file = '../../segmentation/XY_3/tomo1_XY_3_y/connectors/tomo1_XY_3_y_new_rest_good.pkl'

# layers
layers_file =  '../../segmentation/XY_3/tomo1_XY_3_y/layers/new_labels_layers.dat'


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
