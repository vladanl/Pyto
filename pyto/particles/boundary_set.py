"""
Contains class BoundarySet for manipulation of boundaries associated 
with label particle sets.

Label particles are simply particles (segments) extracted from label files.
Boundary particles have boundaries at the same positions as label particles.  

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
            struct=struct, work_path=work_path, catalog_var=catalog_var,
            box_size=box_size, particle_dir=particle_dir, dtype=dtype, 
            pixelsize=pixelsize, fg_value=fg_value, bkg_value=bkg_value)
            
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
        self._use_tomo = True

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
            self._labels = pkl_data
            self._use_tomo = False
            self._morphology = pkl_data.mor
        except AttributeError:
            raise ValueError(
                ("Pickled object from {} doesn't have labels and "
                 + " morphology attributes.").format(path))

    def get_relative_label_coords(self, abs_coords):
        """
        Converts absolute coordinates to relative.

        In case boundaries were read from the entire image (self._use_tomo
        is True), the absolute and relative coordinates are the same. 

        Alternatively, if boundaries were read from a pickle (self._use_tomo
        is False), relative coords are caluclated using ._labels.inset,
        line in LabelSet.

        Reguires self._use_tomo to be set.

        Argument:
          - abs_coords: absolute coordinates

        Returns: relative coordinates
        """

        if len(abs_coords) > 0:

            # check
            if self._use_tomo is None:
                raise ValueError(
                    "Don't know how to calculate relative coords because "
                    + "self._use_tomo is not set.")

            # set relative coordinates
            if self._use_tomo:

                # using omo so relative and absolute coords the same
                relative_coords = abs_coords

            else:

                # using labels
                inset_start = np.array(
                    [one_inset.start for one_inset in self._labels.inset])
                relative_coords = abs_coords - inset_start

        else:
            relative_coords = np.array([])
            
        return relative_coords

    def write_particles(
            self, identifier, labels=None, ids=None, keep_ids=True,
            group_name=None, pixelsize=None, write=True):
        """
        Writes boundary particles as mrc files and optionally boundaries 
        with labels.

        Boundary particles foreground pixel values are set to self.fg_value 
        and background to self.bkg_value.

        Boundary particle file names wo and with boundaries are formed as 
        follows:
            identifier + "_id-" + id + "_label.mrc"
            identifier + "_id-" + id + "_label_bound.mrc"
        where identifier is given by the argument. If keep_ids is True,
        the existing particle ids are used, otherwise particles are labeled 
        from 0 up.

        Bound particles are extracted from _labels attribute. The coordinates
        are determined from _left_corners_* attributes, _centers_* are 
        not used.

        Sets attributes:
          - _particle_paths: label particle file paths

        Arguments:
          - identifier: dataset identifier
          - labels: instance of this class containing labels
          - ids: particle ids or None to use self._ids
          - keep_ids: flag indicating whether the existing particle ids are
          used to form the file names
          - group_name: group name, needed if arg boundary is not None
          - pixelsize: pixel size in nm
          - write: if False, particles are not written
        """

        # check ids
        if ids is None:
            ids = self._ids
        if len(ids) == 0:
            self._particle_paths = []
            return

        # make particle paths
        if keep_ids:
            write_ids = ids
        else:
            write_ids = list(range(n_ids))            
        n_digits = np.log10(write_ids.max()).astype(int) + 1
        particle_names = [
            ('{}_id-{:0' + str(n_digits) + 'd}_bound.mrc').format(
                identifier, id_) for id_ in write_ids]
        particle_label_names = [
            ('{}_id-{:0' + str(n_digits) + 'd}_bound_label.mrc').format(
                identifier, id_) for id_ in write_ids]

        # save paths as attributes
        self._particle_paths = [
            os.path.normpath( os.path.join(self.particle_dir, name) ) 
            for name in particle_names]
        self._particle_label_paths = [
            os.path.normpath( os.path.join(self.particle_dir, name) ) 
            for name in particle_label_names]
        
        # make particle dir if needed
        os.makedirs(self.particle_dir, mode=self.dir_mode, exist_ok=True)
        
        # make slices
        slices_rel = [
            (slice(coords[0], coords[0] + self.box_size),
             slice(coords[1], coords[1] + self.box_size),
             slice(coords[2], coords[2] + self.box_size)) 
            for coords in self._left_corners_rel]
        
        # make slices for boundaries
        if labels is not None:
            label_slices_rel = [
                (slice(coords[0], coords[0] + self.box_size),
                 slice(coords[1], coords[1] + self.box_size),
                 slice(coords[2], coords[2] + self.box_size)) 
                for coords in labels._left_corners_abs]
        else:
            label_slices_rel = slices_rel  # dummy
            
        # write particles
        for sl, label_sl, path, label_path, id_ in zip(
                slices_rel, label_slices_rel, self._particle_paths,
                self._particle_label_paths, ids):
        
            # get particle data
            particle_data = self._labels.useInset(
                inset=sl,  mode=u'relative', useFull=True,
                expand=True, value=0, update=False)
            
            # put code to process particles here

            # use labels foreground and background values
            particle_data = np.where(
                particle_data==id_, self.fg_value,  self.bkg_value)

            # put image data to Labels object and write (wo labels)
            particle = pyto.segmentation.Labels(data=particle_data)
            if write:
                particle.write(
                    file=path, dataType=self.dtype, pixel=pixelsize)

            # add labels to particle image
            if labels is not None:

                # get label ids
                i_labels = labels.struct[group_name].indexed_data
                i_labels = i_labels[i_labels.identifiers == identifier]
                cond = i_labels.boundaries.map(lambda x: id_ in x)
                i_labels_cond = i_labels[cond]
                if i_labels_cond.empty:
                    label_ids = []
                else:
                    label_ids = i_labels_cond['ids'].values

                # get label data
                label_data = labels._labels.useInset(
                    inset=label_sl,  mode=u'absolute', useFull=True,
                    expand=True, value=0, update=False)

                # add labels to particle image
                for l_id in label_ids:
                    particle_data[label_data==l_id] = labels.fg_value

                # put image data to Labels object and write (w boundaries)
                particle = pyto.segmentation.Labels(data=particle_data)
                if write:
                    particle.write(
                        file=label_path, dataType=self.dtype,
                        pixel=pixelsize)
    


