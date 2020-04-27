"""
Contains class LabelSet for manipulation of label particle sets.

Label particles are simply particles extracted from label files. Pixels of 
these images have only two values: forgraund and background.

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

class LabelSet(Set):
    """
    Contains methods to extract and save label (segment) particles, and 
    the data about the extracted particles.

    This class can be used alone, but it is really meant to be used
    together with Set class in the following way:
      
      particles, labels = Set.extract_particles(...)

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
            box_size=None, particle_dir=None,  dir_mod=0o775,
            dtype='int16', fg_value=1, bkg_value=0):
        """
        Saves specified arguments as attributes of the dame name.

        Arguments:
          - struct: structure specific (Groups) object that contains the 
          structures of interest, such as tethers or connectors
          - work_path: absolute path to the analysis work file
          - catalog_var: name of the variable in the catalog files that 
          contains the path to the individual dataset pickles corresponding 
          to the structure specific object defined above (arg struct)
          - box_size: length of particle box in pixels
          - particle_dir: directory where particles are written
          - dtype: particle dtype         
          - dir_mode: mode of the created particle directories
          - fg_value: forground value for label particles
          - bkg_value: background value for label particles
        """

        # init from base class
        super().__init__(
            struct=struct, work_path=work_path, catalog_var=catalog_var, 
            box_size=box_size, particle_dir=particle_dir, dtype=dtype)
            
        # set additional arguments
        self.fg_value = fg_value
        self.bkg_value = bkg_value

        
    ##############################################################
    #
    # Extracting label particles
    #

    def get_coordinates_single(
        self, work, group_name, identifier):
        """
        Extracts particles from one dataset.

        Arguments:
          - work: analysis module (needs to have catalog and catalog_directory
          attributes)
          - group_name: dataset (experiment) group name
          - identifier: dataset (experiment) identifier
          - do_write: flag indicating if particle imagess are written
        """
    
       # get absolute pickle path
        self._pickle_path = self.get_pickle_path(
            work=work, group_name=group_name, identifier=identifier)

        # load pickle containing data and extract data
        #logging.debug("Loading pickle: {}".format(self._pickle_path))
        self.load_pickle(path=self._pickle_path)

        # get ids
        self._ids = self.struct[group_name].getValue(
            name='ids', identifier=identifier)
    
        # get relative center coordinates
        self.get_label_box_centers()
        
    def get_pickle_path(self, work, group_name, identifier):
        """
        Finds and returns the absolute path to the individual dataset pickle 
        corresponding to the dataset (experiment) specified by group_name and 
        identifier args.
        
        Arguments:
          - work: analysis module (needs to have catalog and catalog_directory
          attributes)
          - group_name: dataset (experiment) group name
          - identifier: dataset (experiment) identifier

        Returns: absolute path to the pickle
        """

        # find path
        all_pickle_paths = work.catalog.__getattribute__(self.catalog_var)
        pickle_path = all_pickle_paths[group_name][identifier]

        # make path absolute and normalized
        if not os.path.isabs(pickle_path):
            pickle_path = os.path.join(
                os.path.dirname(self.work_path),
                work.catalog_directory, pickle_path)
        pickle_path = os.path.normpath(pickle_path)
        
        return pickle_path
    
    def load_pickle(self, path):
        """
        Loads pickle specified by arg path.

        The pickle is expected to be of type 
        pyto.scene.segmentation_analysis.SegmentationAnalysis .

        Argument:
          - path: pickle file path

        Sets attributes:
          - _labels: labels attribute of the pickle
          - _morphology: 
        """
        
        # load
        with open(path, 'rb') as pickle_fd:
            try:
                pkl_data = pickle.load(pickle_fd, encoding='latin1')
            except TypeError:
                pkl_data = pickle.load(pickle_fd)
            #logging.debug("Loaded pickle {}".format(path))

        # find and set relevant attributes
        try:
            self._labels = pkl_data.labels
            self._morphology = pkl_data.morphology
        except AttributeError:
            raise ValueError(
                ("Pickled object from {} doesn't have labels and "
                 + " morphology attributes.").format(path))

    def get_label_box_centers(self, ids=None):
        """
        Gets particle centers. 

        Sets attributes:
          - _centers_abs
          - _centers_rel
        """
        
        # check ids
        if ids is None:
            ids = self._ids

        # sanity check
        #shape = np.array(self._labels.data.shape)
        #if (shape < self.box_size).any():
        #    raise ValueError(
        #        "Box size {} is larger than the labels shape {}".format(
        #            self.box_size, shape))
            
        # get centers of labels (no inset) and tomo (with inset)
        if len(ids) > 0:
            self._centers_rel = self._morphology.getCenter(
                segments=self._labels, ids=ids)[ids]
            inset_start = np.array(
                [one_inset.start for one_inset in self._labels.inset])
            self._centers_abs = self._centers_rel + inset_start
        else:
            self._centers_rel = []
            self._centers_abs = []
            
    def set_box_coords(self, left_corners=None, centers=None):
        """
        Sets particle coordinate attributes:
          - _left_corners_abs: absolute left corner coordinates
          - _left_corners_rel: relative left corner coordinates
          - _centers_abs: absolute center coordinates
          - _centers_rel: relative center coordinates

        Only one of the arguments left_corners and centers should be 
        specified. If both are given, arg left_corners is used and
        arg centers is ignored.

        Arguments:
          - left_corners: absolute left corners coordinates
          - centers: absolute center coordinates
        """

        # left corners
        if left_corners is not None:
            self._left_corners_abs = left_corners
            self._left_corners_rel = self.get_relative_label_coords(
                abs_coords = left_corners)

        # centers
        elif centers is not None:
            self._centers_abs = centers
            self._centers_rel = self.get_relative_label_coords(
                abs_coords = centers)

    def get_relative_label_coords(self, abs_coords):
        """
        Converts absolute coordinates to relative (to the inset)

        Argument:
          - abs_coords: absolute coordinates

        Returns: relative coordinates
        """

        if len(abs_coords) > 0:
        
            # inset begin
            inset_start = np.array(
                [one_inset.start for one_inset in self._labels.inset])
        
            # set relative coordinates
            relative_coords = abs_coords - inset_start

        else:
            relative_coords = np.array([])
            
        return relative_coords

    def write_particles(self, identifier, ids=None, keep_ids=True, test=False):
        """
        Writes particles as mrc files.

        Particles foreground pixel values are set to self.fg_value 
        and background to self.bkg_value.

        Particle file names are formed as follows:
            identifier + "_id-" + id + "_label.mrc"
        where identifier is given by the argument. If keep_ids is True,
        the existing particle ids are used, otherwise particles are labeled 
        from 0 up.

        Sets attributes:
          - _particle_paths: particle file paths

        Arguments:
          - identifier: dataset identifier
          - ids: particle ids or None to use self._ids
          - keep_ids: flag indicating whether the existing particle are
          used to form the file names
          - test: if True, particles are not written (for testing)
        """
        
        # check ids
        if ids is None:
            ids = self._ids
        if len(ids) == 0:
            self._particle_paths = []
            return

        # make slices
        slices_rel = [
            (slice(coords[0], coords[0] + self.box_size),
             slice(coords[1], coords[1] + self.box_size),
             slice(coords[2], coords[2] + self.box_size)) 
            for coords in self._left_corners_rel]
        
        # make particle paths
        if keep_ids:
            n_digits = np.log10(ids.max()).astype(int) + 1
            particle_names = [
                ('{}_id-{:0' + str(n_digits) + 'd}_label.mrc').format(
                    identifier, id_) for id_ in ids]
        else:
            n_ids = len(ids)
            n_digits = np.log10(n_ids).astype(int) + 1
            particle_names = [
                ('{}_id-{:0' + str(n_digits) + 'd}.mrc').format(
                    identifier, id_) for id_ in list(range(n_ids))]
        self._particle_paths = [
            os.path.normpath( os.path.join(self.particle_dir, name) ) 
            for name in particle_names]
        
        # make particle dir if needed
        os.makedirs(self.particle_dir, mode=self.dir_mode, exist_ok=True)
        
         # write particles
        for sl, path, id_ in zip(
                slices_rel, self._particle_paths, ids):
        
            # get particle data
            particle_data = self._labels.useInset(
                inset=sl,  mode=u'relative', useFull=True,
                expand=True, value=0, update=False)
            
            # put code to process particles here
            particle_data = np.where(
                particle_data==id_, self.fg_value,  self.bkg_value)
            particle = pyto.segmentation.Labels(data=particle_data)

            # write
            if not test:
                particle.write(file=path, dataType=self.dtype)
    
    def add_data(self, group_name, identifier):
        """
        Adds current data to self.data

        Arguments:
          - group_name: dataset (experiment) group name
          - identifier: dataset (experiment) identifier
        """

        if len(self._ids) == 0: return
        
        # make current table
        curr_data = pd.DataFrame({
            'identifier' : identifier, 'group_name' : group_name,
            'id':self._ids,
            'tomo_path' : self._pickle_path,
            'particle_path' : self._particle_paths,
            'left_corner_x' : self._left_corners_rel[:,0],
            'left_corner_y' : self._left_corners_rel[:,1],
            'left_corner_z' : self._left_corners_rel[:,2]}
        )

        # add to the main table
        if self.data is None:
            self.data = curr_data
        else:
            self.data = self.data.append(curr_data, ignore_index=True)
