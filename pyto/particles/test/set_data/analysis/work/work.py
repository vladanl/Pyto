#!/usr/bin/env python
"""
Analysis-like file

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""

__version__ = "$Revision$"

import pyto

# variables
# Note: catalog.tethers_file is set here, unlike in the real case where
# it is read from the catalog files. Therefore:
#  - variable tethers_file in catalogs is irelevant for module work
#  - to be able to test the 'real' case, the paths given below should be
#  relative to the catalog files and not to this file
catalog = pyto.analysis.Catalog()
catalog.tethers_file = {
    'group_x' : {
        'exp_1' : '../../segmentation/exp_1/connectors/tethers_exp-1.pkl',
        'exp_2' : '../../segmentation/exp_2/connectors/tethers_exp-2.pkl'},
    'group_y' : {
        'exp_3' : '../../segmentation/exp_3/connectors/tethers_exp-3.pkl',
        'exp_4' : '../../segmentation/exp_4/connectors/tethers_exp-4.pkl'}
    }

catalog_directory = '../catalogs'
