"""
Class ColumnsMunc13, specific for the columns munc13 project (2023)

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"


import os
import sys
import pickle
import inspect

import numpy as np
import pandas as pd

import pyto
from pyto.spatial.multi_particle_sets import MultiParticleSets


class ColumnsMunc13(object):
    """
    """

    def __init__(self, **kwargs):
        """Sets attributes from arguments.

        """

        # save all args
        self.save_args(kwargs=kwargs)

    def save_args(self, ignore=[], kwargs=None):
        """Saves calling function arguments as attributes (meant for __init__).

        Argument:
          - ignore: (list) argument names that are not set as attributes
        """
         
        frame = inspect.currentframe().f_back
        args, _, keywords, local_vars = inspect.getargvalues(frame)
        [setattr(self, name, local_vars[name]) for name in args
         if name not in ['self'] + ignore]

        for name, value in kwargs.items():
            setattr(self, name, value)
            
def resolve_particle_set(set_name, coloc_case, psets_module=None):
    """Resolves the specified particle set name used in colocalization.

    Particle set conversion rules depend on the coloclization case 
    (arg coloc_case) and are specified in the docs for the corresponding
    function (such as resolve_particle_set_pick2_bin2_xd5_v4).

    If arg psets_module is None, finds the class (set) name and number that 
    correspond to the specified (arg) set_name. 

    If arg psets_module is specified, in addition to the above, finds  the path 
    to the multi particle sets (pyto.spatial.MultiParticleSets) pickle that
    contains the particle set specified by arg set_name and reads the pickle.

    Arguments:
      - set_name: colocalization-like particle set name
      - coloc_case: coloalization case
      - module: (loaded module or a path) module that contains paths to 
      pickled MultiParticleSets

    Returns:
      - mps: (MultiParticlesets) multi particle sets object read from the 
      pickle path that contains the specified set and may contain other sets, 
      or None if arg psets_module is None
      - class_name: class name as in mps.particles column mps.class_name_col
      - class_number: class number as in mps.particles column 
      mps.class_number_col
    """

    if coloc_case == 'pick-2_bin-2_xd-5_v4':
        res = resolve_particle_set_pick2_bin2_xd5_v4(
            set_name=set_name, psets_module=psets_module)
        return res
    else:
        raise ValueError(
            f"Argument coloc_case {coloc_case} was not understood. Currently "
            + "implemented cases are 'pick-2_bin-2_xd-5_v4'")
    
def resolve_particle_set_pick2_bin2_xd5_v4(set_name, psets_module=None):
    """Resolves the specified particle set name used in colocalization.

    Does the following particle set name conversion:

      Colocalization names      multi particle set name, number
        'pre'                       'pre', None
        'preapN'                    'pre', [N]
        'pre*'                      'pre', [one or more ints]
        'post'                      'post', None
        'postapN'                   'post', [N]
        'post*'                     'post', [one or more ints]
        'tetmonoc'                  'nocentroids', None
        'tetmoc'                    'tethers, None
        'tetmoc6'                   'tethers6, None
        'tethiall'                  'tet', None
        'tethishort'                'tet', [0]
        'tethimed'                  'tet', [1]
        'tethilong'                 'tet', [2]

    Arguments:
      - set_name: colocalization-like particle set name
      - module: (loaded module or a path) module that contains paths to 
      pickled MultiParticleSets

    Returns:
      - mps: (MultiParticlesets) multi particle sets object read from the 
      pickle path that contains the specified set and may contain other sets, 
      or None if arg psets_module is None
      - class_name: class name as in mps.particles column mps.class_name_col
      - class_number: class number as in mps.particles column 
      mps.class_number_col
    """

    pre_post_names = {
        # commented out because pre particles contains only these classes 
        #'pre': [0,  1,  2,  3,  5,  6,  7,  8,  9, 11, 12, 14, 16, 17, 19, 20,
        #      21, 22, 23, 24, 25, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
        #      38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54],
        'pre0': [41],
        'pre1': [42],
        'pre2': [38, 40, 43, 44, 51, 53],
        'pre3': [7, 22, 23, 24, 41, 42, 48],
        
        # commented out because post particles contains only these classes 
        #'post': [0,  1,  2,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,
        #       17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 29, 31, 34, 35, 36,
        #       37, 38, 39, 40, 41, 42, 43, 47, 48, 49, 52, 53, 54, 55, 56, 57,
        #       58, 60, 62, 67, 70, 71, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84,
        #       85, 86, 87, 88],
        'post0': [23, 25],
        'post1': [25],
        'post2': [6, 10],
        'post3': [41],
        'post4': [41, 42, 43],
        'post5': [42],
        'post6': [43],
        'post7': [54],
        'post8': [55],
        'post9': [58],
        'post10': [62],
        'post11': [83, 85],
        'post12': [4, 12, 17],
        'post13': [12, 14, 17]}
    
    # pre and post
    for class_name in ['pre', 'post']:
        
        if set_name.startswith(class_name):

            # get MultiParticleSets instance
            if psets_module is not None:
                mps_path = getattr(psets_module, f"{class_name}_mps_path") 
                if not os.path.isabs(mps_path):
                    psets_module_dir = os.path.dirname(psets_module.__file__)
                    mps_path = os.path.normpath(
                        os.path.join(psets_module_dir, mps_path))
                    mps = MultiParticleSets.read(mps_path, verbose=False)
            else:
                mps = None

            # get class description
            class_desc = set_name.removeprefix(class_name)

            # get class number
            if len(class_desc) == 0:
                class_num = None
            elif set_name.startswith('ap'):
                class_num = set_name.removeprefix('ap')
                class_num = [int(class_num)]
            else:
                class_num = pre_post_names[set_name]

            return mps, class_name, class_num

    # morse tethers 
    if set_name.startswith('tetmo'):
        cl_names = {'tetmonoc': 'nocentroids', 'tetmoc': 'tethers',
                    'tetmoc6': 'tethers6'}
        if psets_module is not None:
            mps_path = getattr(psets_module, 'tethers_morse_mps_path')
            if not os.path.isabs(mps_path):
                psets_module_dir = os.path.dirname(psets_module.__file__)
                mps_path = os.path.normpath(
                    os.path.join(psets_module_dir, mps_path))
                mps = MultiParticleSets.read(mps_path, verbose=False)
        else:
            mps = None
        class_name = cl_names[set_name]
        class_num = None
        return mps, class_name, class_num

    # pyto tethers
    if set_name.startswith('tethi'):
        if psets_module is not None:
            cl_nums = {'tethiall': None, 'tethishort': 0, 'tethimed': 1,
                       'tethilong': 2}
            mps_path = getattr(psets_module, 'tethers_hiconnect_mps_path')
            if not os.path.isabs(mps_path):
                psets_module_dir = os.path.dirname(psets_module.__file__)
                mps_path = os.path.normpath(
                    os.path.join(psets_module_dir, mps_path))
                mps = MultiParticleSets.read(mps_path, verbose=False)
        else:
            mps = None
        class_name = 'tethi'
        class_num = cl_nums[set_name]
        return mps, class_name, [class_num]
    
    raise ValueError(
        f"Set {set_name} was not understood. Currently existing sets "
        "are 'pre', 'post', 'tetmo' and 'tethi'.")

def get_group(tomo_id):
    """Returns group name of the specified tomo

    """
    group, number = tomo_id.rsplit('_', 1)
    return group
