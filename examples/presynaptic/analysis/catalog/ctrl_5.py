"""
Segmentation and analysis meta-data
"""
from __future__ import unicode_literals


#######################################################
#
# Experiment description and file names
#  - required except when noted explicitly
#

# identifier
identifier = 'ctrl_5'

# experimental condition
treatment = 'ctrl'

# further info (optional)
animal_nr = 'ctrl 4'

# synaptic vesicles
sv_file = '../../segmentation/cell_tomo-7/vesicles/tomo_vesicles.pkl'
sv_membrane_file = '../../segmentation/cell_tomo-7/vesicles/tomo_mem.pkl'
sv_lumen_file = '../../segmentation/cell_tomo-7/vesicles/tomo_lum.pkl'

# hierarchical segmentation of tethers and connectors
tethers_file = '../../segmentation/cell_tomo-7/connectors/tomo_new_tethers_good.pkl'
connectors_file = '../../segmentation/cell_tomo-7/connectors/tomo_new_connectors_good.pkl'

# clustering (optional)
cluster_file = '../../segmentation/cell_tomo-7/cluster/tomo_new_connectors_good_cluster.pkl'

# layers
layers_file =  '../../segmentation/cell_tomo-7/layers/labels_layers.dat'


########################################################
#
# Observations (all optional)
#

# mitochondria in the presyn terminal
mitochondria = True


######################################################
#
# Microscopy
#    - pixel_size required, all other optional
#

# microscope
microscope = 'polara_1'

# pixel size [nm]
pixel_size = 1.888

# person who recorded the series
operator = 'Someone'

# person who did membrane segmentation
segmented = 'Someone else'
