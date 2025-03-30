"""
Contains class SetPath for manipulation of paths related to particle sets

# Author: Vladan Lucic (Max Planck Institute for Biochemistry)
# $Id$
"""
#from __future__ import unicode_literals
#from __future__ import division
#from builtins import zip
#from past.utils import old_div

__version__ = "$Revision$"

import os
import sys
import subprocess
import importlib
import pickle
import warnings

import numpy as np
import scipy as sp
try:
    import pandas as pd
except ImportError:
    pass # Python 2
import pyto

class SetPath(object):
    """Manipulation of paths related to particle sets.

    Methods:
      - get_pickle_path(): gets a structure pickle of an individual dataset 
      (tomo) 
      - get_tomo_info_path(): 
      - import_tomo_info()
      - get_tomo_path(): finds tomo path from imported tomo_info file
   """

    ##############################################################
    #
    # Initialization
    #

    def __init__(self, catalog_var=None, helper_path=None, common=None):
        """
        Sets variables.

        Arguments:
          - catalog_var: name of the attribute of a structure specific 
          object that contains path to an individual dataset (tomogram) 
          pickle, it is the same as the name of a variable defined in a 
          catalog file.
          - helper_path: a path used to convert paths in this instance 
          (see convert_path() doc), default value os.pathgetcwd()
          - common: common pattern used to convert paths in this instance 
          (see convert_path() doc)
        """

        # set attributes from arguments
        self.catalog_var = catalog_var
                
        # relative directory of the tomo info file in respect to individual
        # dataset pickles
        self.tomo_info_dir_relative = '../common'

        # tomo info file base (without directory)
        self.tomo_info_base = 'tomo_info'

        # name of the variable in a tomo_info file that contains the name
        # of the tomo file
        self.tomo_path_var = 'image_file_name'

        # set vatiable for path conversion
        self.helper_path = helper_path
        if helper_path is None:
            self.helper_path = os.getcwd()
        self.common = common

    ##############################################################
    #
    # Mathods
    #

    def convert_path(self, path):
        """Converts a beginning part of arg  path.

        Typically used to convert paths between different file 
        system organizations.

        Given common and helper strings, finds common in (arg) path 
        and in helper. Then it replaces everything before and including 
        common by the path by everything before and including common in
        helper.

        The common and helper strings are specified by self.common and
        self.helper_path.

        For example, if the path in the old file system is:
          path = /pre_old/.../common/post/.../name.ext
        and:
          common = 'common'
          helper_path = '/pre_new/.../common/whatever_else/.../other'
        executing:
          sp = SetPath(common=common, helper_path=helper_path)
          new_path = sp.convert_path(path)
        results in:
          new_path = /pre_new/.../common/post/.../name.ext

        Requires self.common and self.helper_path to be set at initialization.
        If self.common is None, returns the unchanged arg path.

        Pattern self.common has to be present exactly one time in both path 
        and self.helper_path. Otherwise, ValueError is raised.

        Argument:
          - path: path to be converted

        Returns: converted path
        """

        if self.common is not None:
            h_path_split = self.helper_path.split(self.common)
            path_split = path.split(self.common)
            if (len(h_path_split) != 2) or (len(path_split) != 2):
                raise ValueError(
                    f'Both {path} and {self.helper_path} have to contain '
                    f'{self.common} exactly once.')
            converted = h_path_split[0] + self.common + path_split[1]

        else:
            converted = path
            
        return converted

    def get_pickle_path(
            self, group_name, identifier, struct=None, catalog_dir=None,
            work=None, work_path=None, old_dir=None, new_dir=None):
        """
        Finds and returns the absolute path to the individual dataset pickle 
        that corresponds to the dataset (experiment) specified by args 
        group_name and identifier, and to the structures defined by 
        self.catalog_var. 

        The pickle path is determined in the following way:
          - If arg struct is specified, pickle path is read from the 
          self.catalog_var property of struct for the specified group
          and experiment
          - Otherwise, arg work has to be specified, pickle path is read 
          from the self.catalog_var property of work.catalog for the 
          specified group and experiment
          - If the pickle path determined above is a relative path, it is 
          converted to relative usinf the first possible among:
            - catalog_dir / (relative) pickle_path 
            - (absolute) work.catalog_directory / (relative) pickle_path
            - work_path / (relative) work.catalog_directory / (relative) 
              pickle_path
          - Finally, the (arg) old_dir part of the (absolute) pickle path
          is replaced by the (arg) new_path, if the two args are specified
        
        Arguments:
          - group_name: dataset (experiment) group name
          - identifier: dataset (experiment) identifier
          - struct: structure specific object (loaded from a structure 
          specific pickle)
          - catalog_dir: absolute catalog directory
          - work: analysis module (needs to have catalog and catalog_directory
          attributes)
          - work_path: absolute path to work file (only the dir part is used) 
          - old_dir: absolute directory prefix of the pickle path that should
          be substituted by arg new_dir
          - new_dir: absolute directory used for the substitution

        Returns: absolute path to the pickle
        """

        # find path
        if struct is not None:

            # based on structure spicific pickle
            pickle_path = struct[group_name].getValue(
                identifier=identifier, name=self.catalog_var)

        elif work is not None:

            # based on work 
            all_pickle_paths = work.catalog.__getattribute__(self.catalog_var)
            pickle_path = all_pickle_paths[group_name][identifier]
            
        else:
            raise ValueError("Argument struct or work has to be specified")
            
        # make path absolute and normalized
        if not os.path.isabs(pickle_path):
            if catalog_dir is not None:
                pickle_path = os.path.join(catalog_dir, pickle_path)
            elif (work_path is not None) and (work is not None):
                if os.path.isabs(work.catalog_directory):
                    pickle_path = os.path.join(
                        work.catalog_directory, pickle_path)
                else:
                    pickle_path = os.path.join(
                        os.path.dirname(work_path),
                        work.catalog_directory, pickle_path)
        pickle_path = os.path.normpath(pickle_path)

        # replace part of the (absolute) pickle_path
        if (old_dir is not None) and (new_dir is not None):
            if not os.path.isabs(old_dir):
                raise ValueError(
                    "Arg old_dir havs to be absolute")
            pickle_path = os.path.join(
                new_dir, os.path.relpath(pickle_path, old_dir))
        
        return pickle_path

    def get_tomo_info_path(self, pickle_path):
        """
        Returns the path to the tomo info file.

        It is determined in the following way:
          - from the pickle file, finds the location of the tomo info file
          for that dataset (based on self.tomo_info_dir_relative and
          self.tomo_info_base)

        Argument:
          - pickle_path: absolute path to an individual dataset pickle file

        Returns: normalized path to the tomogramtomo info file
        """

        # try to find tomo_info file
        pickle_dir, _ = os.path.split(pickle_path)
        tomo_info_path = os.path.normpath(
            os.path.join(
                pickle_dir, self.tomo_info_dir_relative,
                self.tomo_info_base+'.py'))

        return tomo_info_path

    def import_tomo_info(self, tomo_info_path):
        """
        Imports tomo info file

        Argument:
          - tomo_info_path: tomo info path

        Returns: tomo info module
        """

        spec = importlib.util.spec_from_file_location(
            self.tomo_info_base, tomo_info_path)
        tomo_info = spec.loader.load_module(spec.name)

        return tomo_info

    def get_tomo_path(self, pickle_path):
        """
        Guesses the path to the tomo file from one of the pickle file names
        in the following way:
          - from the pickle file, finds the location of the tomo info file
          for that dataset (based on self.tomo_info_dir_relative and
          self.tomo_info_base)
          - loads tomo info file
          - reads tomo path from the tomo info file (variable
          self.tomo_path_var)

        That is, for the default values of the attributes mentioned above,
        requires the "standard" file organization:
            base dir /
              common /
                tomo_info.py
              pickle dir /
                pickle file

        Sets attributes:
          - tomo_info_path: tomo info path
          - tomo_info: tomo info module

        Argument:
          - pickle_path: absolute path to an individual dataset pickle file

        Returns: normalized path to the tomogram
        """

        # try to find tomo_info file
        self.tomo_info_path = self.get_tomo_info_path(pickle_path=pickle_path)
        
        # import tomo info file
        self.tomo_info = self.import_tomo_info(
            tomo_info_path=self.tomo_info_path)

        # get tomo path
        tomo_path_from_info = self.tomo_info.__getattribute__(
            self.tomo_path_var)
        if os.path.isabs(tomo_path_from_info):
            tomo_path = tomo_path_from_info
        else:
            tomo_info_dir = os.path.split(self.tomo_info_path)[0]
            tomo_path = os.path.join(tomo_info_dir, tomo_path_from_info)

        return os.path.normpath(tomo_path)

