#!/usr/bin/env python
"""
Allows batch modifications and execution of single dataset segmentation
and analysis scripts.

Performs one or more of the following tasks:
  - Modifies multipe single dataset scripts according to the specified rules
  - Batch executes the modified or previously existing single dataset scripts
  - Modifies multiple catalog files (contain result file paths and other
  metadata).

For example, this script can:
  - Modifies parameters in existing single dataset segmentation and
  analysis scripts) and (if specified) saves then under different names
- Executes the modufied scripts
- Adds the information about new resut files (paths) obtained by the
script execution from the orevious step and adds or modifies other metadata

$Id$
# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
"""
from __future__ import unicode_literals
#from builtins import str
from builtins import range

__version__ = "$Revision$"


import sys
import os
import re
import logging
import pickle
import runpy
import imp

import numpy
import scipy

import pyto

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s %(module)s.%(funcName)s():%(lineno)d %(message)s',
    datefmt='%d %b %Y %H:%M:%S')


##############################################################
#
# Parameters (need to be edited)
#
##############################################################

##############################################################
#
# General
#

# Define single dataset directories containing scripts that need to be
# modified or executed
directories = (['ctrl-mouse_' + str(ind) for ind in range(1,8)]
               + ['CAM_' + str(ind) for ind in range(1,8)]
               + ['ctrl-rat_' + str(ind) for ind in range(1,6)]
               + ['egta_0' + sym for sym in ['a','b','c','d','e','f']]
               + ['egta_1' + sym for sym in ['a','b','c','d','e']]
               + ['egta_2']
               + ['egta_3' + sym for sym in ['a','b','c','d','e','f']])
directories_discard = ['syncam_ox_4', 'syncam_ox_7']
directories = [ident for ident in directories
               if ident not in directories_discard]

# List of the single dataset script file paths
scripts = ['../../all_tomograms/' + dir + '/cleft/cleft.py'
           for dir in directories]

# List of single dataset (experiment) identifiers (used only for catalogs)
identifiers = (['syncam_ox_ctrl_' + str(ind) for ind in range(1,8)]
               + ['syncam_ox_' + str(ind) for ind in range(1,8)]
               + ['egta_ctrl_' + str(ind) for ind in range(1,6)]
               + ['egta_0' + sym for sym in ['a','b','c','d','e','f']]
               + ['egta_1' + sym for sym in ['a','b','c','d','e']]
               + ['egta_2']
               + ['egta_3' + sym for sym in ['a','b','c','d','e','f']])
identifiers_discard = ['egta_ctrl_6']
identifiers = [ident for ident in identifiers
               if ident not in identifiers_discard]

##############################################################
#
# Modifying scripts
#

# flag indicating if the scripts need to be modified
modify_scripts = True

# script name (path) pattern that is changed for the modified scripts
old_script_pattern = r'cleft\.py$'

# replacements for the script name pattern (above)
new_script_replace = 'cleft.py'

# Rules for modifying scripts, all rules are applied to each line (from each
# script), two forms are available:
#   - line_matching_pattern : new_line; each line that matches pattern
#   is replaced by the specified line (new_line needs to end with \n).
#   Example: r'^n_layers =' : 'n_layers = 4' + os.linesep
#   - line_matching_pattern : {pattern, pattern_replacement}; in each line
#   that matches line matching pattern, pattern is replaced by
#   pattern_replacement and the rest of the line is left unchanged
#   Example: r'^lay_suffix =' : {r'_layers\.em' : '_layers-4.em'}
script_rules = {
    r'^n_layers =': 'n_layers = 4' + os.linesep,
    r'^n_extra_layers =': 'n_extra_layers = 2' + os.linesep,
    r'^membrane_thickness': 'membrane_thickness = 1' + os.linesep,
    r'^layers_only =': 'layers_only = True' + os.linesep,
    r'^lay_suffix =': {r'_layers\.em': '_layers-4.em'},
    r'^cleft_res_suffix =': {r'_cl\.dat': '_cl_layer-4.dat'},
    r'^cl_suffix =': {r'_cl\.pkl': '_cl_layer-4.pkl'}
    }

##############################################################
#
# Executing scripts
#

# flag indicating if scripts need to be run
run_scripts = True

##############################################################
#
# Modifying catalogs
#

# flag indicating if catalogs need to be modified
modify_catalogs = True

# catalog directory
catalog_dir = '../catalog/'

# list of catalog file names or name patterns
#catalogs  = [r'[^_].*\.py$']   # extension 'py', doesn't start with '_'
#catalogs = [r'^syncam_ox_1.py$']  # only syncam_ox_1.py
catalogs = [ident + '.py$' for ident in identifiers]  # based on identifiers

# catalog lines to copy, each matched lines is first left unmodified and then,
# on the next line, copied and modified by the catalog_rules
catalog_copy = [r'^cleft_layers =']

# rules for modifying lines of catalogs, same format as script_rules
catalog_rules = {
    r'^cleft_': {r'cleft_vl': 'cleft'},
    r'^cleft_layers =': {r'_cl': '_layers-thin'},
    r'^cleft_layers_4 =': {r'_cl': ''},
    r'^cleft_segmentation_hi_layers =': {r'^cleft': '#cleft'},
    r'^cleft_segmentation_hi_layers_4 = : {r'_cl': ''}}


################################################################
#
# Main function (edit if you know what you're doing)
#
###############################################################

def main():

    # write and run scripts
    for script_path in scripts:

        if modify_scripts:

            # open old and new scripts
            new_path = re.sub(
                old_script_pattern, new_script_replace, script_path)

            # write new (modified) script
            logging.info("Modifying script " + new_path)
            pyto.util.bulk.replace(
                old=script_path, new=new_path, rules=script_rules)

        else:

            # use existing script
            new_path = script_path

        # run script
        if run_scripts:
            logging.info("Runninging script " + new_path)
            pyto.util.bulk.run_path(path=new_path)

    # add entries to catalogs
    if modify_catalogs:

        # read directory
        # Note: can not put in the for loop statement below, because the
        # loop may modify the directory
        all_files = os.listdir(catalog_dir)

        # modify all catalogs
        for cat in all_files:

            # check all catalog patterns
            for cat_pattern in catalogs:

                # do all modifications for the current catalog
                if re.search(cat_pattern, cat) is not None:
                    cat_path = os.path.join(catalog_dir, cat)
                    logging.info("Modifying catalog " + cat_path)
                    pyto.util.bulk.replace(
                        old=cat_path, rules=catalog_rules,
                        repeat=catalog_copy)
                    break


# run if standalone
if __name__ == '__main__':
    main()
