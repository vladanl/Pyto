"""
Contains class BoundarySet for manipulation of boundaries associated 
with label particle sets.

Label particles are simply particles (segments) extracted from label files.
Boundary particles show boundaries at the same positions as label particles.  

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
from __future__ import unicode_literals
from __future__ import division
#from builtins import zip
from past.utils import old_div

__version__ = "$Revision$"

import os
import sys
import subprocess
import importlib
import pickle
import logging

import numpy as np
import scipy as sp
try:
    import pandas as pd 
except ImportError:
    pass # Python 2
import pyto
from .set import Set
from .label_set import LabelSet

class BoundarySet(LabelSet):
    """
    Contains methods to extract and save boundaries corresponding
    label (segment) particles.

    This class can is meant to be used together with Set and LabelSet 
    classes in the following way:
      
       particles, labels, boundaries = Set.extract_particles(...)

    All instance methods are used for Set.extract_particles()

    Important attributes:
      - data: Pandas.DataFrame containing all important data. Columns are
      'identifier', 'group_name', 'id', 'tomo_path', 'particle_path',
      'left_corner_x', 'left_corner_y', 'left_corner_z'. Each row corresponds 
      to one patricle (one id of one dataset) 
      - all arguments passed to the constructor are saved as attributes
      of the same name
      - attributes starting with '_' (_tomo, _tomo_path, ...) contain
      values for the current dataset (one of the datasets in case
      several datasets are processed)
    """

    ##############################################################
    #
    # Initialization
    #

    def __init__(
            self, struct=None, work_path=None, catalog_var=None, 
            box_size=None, particle_dir=None, dir_mod=0o775, dtype='int16', 
            pixelsize=None, fg_value=1, bkg_value=0):
        """
        Saves specified arguments as attributes of the same name.

        Arguments:
          - struct: structure specific (Groups) object that contains the 
          structures of interest, such as tethers or connectors
          - work_path: absolute path to the analysis work file
          - catalog_var: name of the variable in the catalog files that 
          contains the path to the individual dataset pickles corresponding 
          to the structure specific object (segments) defined above (arg struct)
          - boundary_var: name of the variable in the catalog files that 
          contains the path to the individual dataset pickles corresponding 
          - box_size: length of particle box in pixels
          - particle_dir: directory where particles are written
          - dtype: particle dtype         
          - dir_mode: mode of the created particle directories
          - pixelsize: pixel size in nm
          - fg_value: forground value for label particles
          - bkg_value: background value for label particles
        """

        # init from base class
        super().__init__(
            struct=struct, work_path=work_path, box_size=box_size,
            particle_dir=particle_dir, dtype=dtype, pixelsize=pixelsize,
            fg_value=fg_value, bkg_value=bkg_value)
            
        # set arguments from arguments
        #self.catalog_var = catalog_var  # currently not used

        # name of the variable in a tomo_info file that contains the name
        # of the tomo file 
        self.tomo_path_var = 'labels_file_name'
        
        
    ##############################################################
    #
    # Extracting label particles
    #

    def read_tomo(self, work, group_name, identifier, label_set):
        """
        Reads boundary image for one dataset.

        Uses get_tomo_path() to guess the boundary image path. Consequently,
        requires the "standard" file organization:
            base dir /
              common /
                tomo_info.py
              pickle dir /
                pickle file
            
        Arguments:
          - work: analysis module (needs to have catalog and catalog_directory
          attributes)
          - group_name: dataset (experiment) group name
          - identifier: dataset (experiment) identifier
          - label_set: LabelSet object corresponding to the current dataset

        Sets attributes:
          - _tomo_path:
          - _tomo:
        """
        # get tomo path
        self._tomo_path = self.get_tomo_path(pickle_path=label_set._pickle_path)

        # read tomo
        self._tomo = pyto.grey.Image.read(
            file=self._tomo_path, header=True, memmap=True)

    def get_relative_label_coords(self, abs_coords):
        """
        Converts absolute coordinates to relative, but because boundaries
        are read from the entire image, the absolute and relative 
        coordinates are the same. 

        In other words, returns the argument. Needed because this class
        inherits from LabelSet, where relative coords differe from the 
        absolute.

        Argument:
          - abs_coords: absolute coordinates

        Returns: relative coordinates
        """

        if len(abs_coords) > 0:
        
           # set relative coordinates
            relative_coords = abs_coords

        else:
            relative_coords = np.array([])
            
        return relative_coords


