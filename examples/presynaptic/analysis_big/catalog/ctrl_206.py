"""
Segmentation and analysis meta-data
"""


#######################################################
#
# Experiment description and file names
#

# identifier
identifier = 'ctrl_206'

# experimental condition
treatment = 'ctrl'

# batch and original identifier
batch = 'XY_2'

ori_identifier = 'tomo6_XY_2_ctrl'

# synaptic vesicles
sv_file = '../../segmentation/XY_2/tomo6_XY_2_ctrl/vesicles/tomo6_XY_2_ctrl_vesicles.pkl'
sv_membrane_file = '../../segmentation/XY_2/tomo6_XY_2_ctrl/vesicles/tomo6_XY_2_ctrl_mem.pkl'
sv_lumen_file = '../../segmentation/XY_2/tomo6_XY_2_ctrl/vesicles/tomo6_XY_2_ctrl_lum.pkl'

# hierarchical segmentation of tethers and connectors
tethers_file = '../../segmentation/XY_2/tomo6_XY_2_ctrl/connectors/tomo6_XY_2_ctrl_new_AZ_good.pkl'
connectors_file = '../../segmentation/XY_2/tomo6_XY_2_ctrl/connectors/tomo6_XY_2_ctrl_new_rest_good.pkl'

# layers
layers_file =  '../../segmentation/XY_2/tomo6_XY_2_ctrl/layers/balls-1_layers.dat'


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
