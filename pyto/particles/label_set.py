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
        Saves specified arguments as attributes of the dame name.

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
            particle_dir=particle_dir, dtype=dtype, pixelsize=pixelsize)
            
        # set additional arguments
        self.catalog_var = catalog_var
        self.fg_value = fg_value
        self.bkg_value = bkg_value

        
    ##############################################################
    #
    # Extracting label particles
    #

    def get_coordinates_single(
        self, work, group_name, identifier):
        """
        Determines absolute and relative label center coordinates.

        The following steps are performed:
        
        Reads segmented image for one dataset and the coordinates of 
        individual segments.

        1) Figures out the path to the individual dataset pickle for 
        the dataset specified by args group_name and identifier, and
        for structures defined by self.catalog_var.

        2) Loads the pickle, extracts labes and morphology attributes.

        3) Determines relative label coordinates of all segments and uses
        inset to convert them to absolute label center coordinates 
        (meant for structures like connectors and tethers).

        Arguments:
          - work: analysis module (needs to have catalog and catalog_directory
          attributes)
          - group_name: dataset (experiment) group name
          - identifier: dataset (experiment) identifier

        Sets attributes:
          - _pickle_path
          - _ids: 
          - _boundary_ids
          - _labels: labels attribute of the pickle
          - _morphology: 
          - _centers_abs
          - _centers_rel
        """
    
       # get absolute pickle path
        self._pickle_path = self.get_pickle_path(
            work=work, group_name=group_name, identifier=identifier)
        #print(self._pickle_path)

        # load pickle containing data and extract data
        #logging.debug("Loading pickle: {}".format(self._pickle_path))
        self.load_pickle(path=self._pickle_path)

        # get ids
        self._ids = self.struct[group_name].getValue(
            name='ids', identifier=identifier)

        # get boundary ids (not needed for this method)
        try:
            self._boundary_ids = self.struct[group_name].getValue(
                name='boundaries', identifier=identifier)
        except AttributeError:
            self._boundary_ids = None
            
        # get relative center coordinates
        self.get_label_box_centers()
        
    def get_pickle_path(self, work, group_name, identifier):
        """
        Moving to SetPath

        Finds and returns the absolute path to the individual dataset pickle 
        that corresponds to the dataset (experiment) specified by args 
        group_name and identifier, and to the structures defined by 
        self.catalog_var.
        
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
        #if self.mode == 'segment':
        try:
            self._labels = pkl_data.labels
            self._morphology = pkl_data.morphology
        except AttributeError:
            raise ValueError(
                ("Pickled object from {} doesn't have labels and "
                 + " morphology attributes.").format(path))
        #elif self.mode == 'boundary':
        #    self._labels = pkl_data
        #    print('load: pkl_data.inset {}'.format(pkl_data.inset))
        #    print('load: pkl_data.data.shape {}'.format(pkl_data.data.shape))
            #self._morphology = pkl_data.mor  # commented out cause not needed
        #else:
        #    raise ValueError(
        #        "Attribut mode {}".format(mode)
        #        + " has to be 'segment', or 'boundary'.") 

    def get_label_box_centers(self, ids=None):
        """
        Gets particle centers (relative and absolute). 

        Sets attributes:
          - _centers_abs: absolute center coordinates that take image inset
          into account (as in the greyscale tomo) 
          - _centers_rel: center coordinates that don't take image inset
          into account (obtained directly from labels array as stored in 
          the dataset pickle. 
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
            
        # get relative (this data) and absolute (tomo) label centers 
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
        Converts from absolute to relative coordinates.

        Only one of the arguments left_corners and centers should be 
        specified. If both are given, arg left_corners is used and
        arg centers is ignored.

        Sets particle coordinate attributes (n_particles, n_dims ndarrays),
        dependig on the arguments:
          - _left_corners_abs: absolute left corner coordinates
          - _left_corners_rel: relative left corner coordinates
          - _centers_abs: absolute center coordinates
          - _centers_rel: relative center coordinates
        If arg left_corners is specified, _left_corners_* attributes are
        set and _centers_* attributes are left unchanged. Alternatively,
        if arg centers is specified, _centers_* are set and _left_corners_*
        are left unchanged.

        Arguments:
          - left_corners: absolute left corners coordinates
          - centers: absolute center coordinates
        """

        # left corners
        if left_corners is not None:
            self._left_corners_abs = left_corners
            self._left_corners_rel = self.get_relative_label_coords(
                abs_coords=left_corners)

        # centers
        elif centers is not None:
            self._centers_abs = centers
            self._centers_rel = self.get_relative_label_coords(
                abs_coords=centers)

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

    def write_particles(
            self, identifier, boundary=None, ids=None, keep_ids=True,
            group_name=None, pixelsize=None, write=True):
        """
        Writes label particles as mrc files and optionally labels with 
        boundaries.

        Label particles foreground pixel values are set to self.fg_value 
        and background to self.bkg_value.

        Label particle file names wo and with boundaries are formed as follows:
            identifier + "_id-" + id + "_label.mrc"
            identifier + "_id-" + id + "_label_bound.mrc"
        where identifier is given by the argument. If keep_ids is True,
        the existing particle ids are used, otherwise particles are labeled 
        from 0 up.

        Label particles are extracted from _labels attribute. The coordinates
        are determined from _left_corners_* attributes, _centers_* are 
        not used.

        Sets attributes:
          - _particle_paths: label particle file paths

        Arguments:
          - identifier: dataset identifier
          - boundary: instance of this class containing boundaries
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
            ('{}_id-{:0' + str(n_digits) + 'd}_label.mrc').format(
                identifier, id_) for id_ in write_ids]
        particle_bound_names = [
            ('{}_id-{:0' + str(n_digits) + 'd}_label_bound.mrc').format(
                identifier, id_) for id_ in write_ids]

        # save paths as attributes
        self._particle_paths = [
            os.path.normpath( os.path.join(self.particle_dir, name) ) 
            for name in particle_names]
        self._particle_bound_paths = [
            os.path.normpath( os.path.join(self.particle_dir, name) ) 
            for name in particle_bound_names]
        
        # make particle dir if needed
        os.makedirs(self.particle_dir, mode=self.dir_mode, exist_ok=True)
        
        # make slices
        slices_rel = [
            (slice(coords[0], coords[0] + self.box_size),
             slice(coords[1], coords[1] + self.box_size),
             slice(coords[2], coords[2] + self.box_size)) 
            for coords in self._left_corners_rel]
        
        # make slices for boundaries
        if boundary is not None:
            bound_slices_rel = [
                (slice(coords[0], coords[0] + self.box_size),
                 slice(coords[1], coords[1] + self.box_size),
                 slice(coords[2], coords[2] + self.box_size)) 
                for coords in boundary._left_corners_abs]
        else:
            bound_slices_rel = slices_rel  # dummy
            
        # write particles
        for sl, bound_sl, path, bound_path, id_ in zip(
                slices_rel, bound_slices_rel, self._particle_paths,
                self._particle_bound_paths, ids):
        
            # get particle data
            particle_data = self._labels.useInset(
                inset=sl,  mode=u'relative', useFull=True,
                expand=True, value=0, update=False)
            
            # put code to process particles here

            # use labels foreground and background values
            particle_data = np.where(
                particle_data==id_, self.fg_value,  self.bkg_value)

            # put image data to Labels object and write (wo boundaries)
            particle = pyto.segmentation.Labels(data=particle_data)
            if write:
                particle.write(
                    file=path, dataType=self.dtype, pixel=pixelsize)

            # add boundaries to particle image
            if boundary is not None:

                # get boundary ids
                bound_ids = boundary.struct[group_name].getValue(
                    identifier=identifier, name='boundaries', ids=id_)

                # get boundary data
                bound_data = boundary._tomo.useInset(
                    inset=bound_sl,  mode=u'absolute', useFull=True,
                    expand=True, value=0, update=False)

                # add boundaries to particle image
                for b_id in bound_ids:
                    particle_data[bound_data==b_id] = boundary.fg_value

                # put image data to Labels object and write (w boundaries)
                particle = pyto.segmentation.Labels(data=particle_data)
                if write:
                    particle.write(
                        file=bound_path, dataType=self.dtype,
                        pixel=pixelsize)
    
    def add_data(self, group_name, identifier):
        """
        Adds current data to self.metadata and self.data pandas tables.

        Doesn't do anything if this object has no ids (self._ids).

        Arguments:
          - group_name: dataset (experiment) group name
          - identifier: dataset (experiment) identifier
        """

        if len(self._ids) == 0: return
        
        # make current metatable
        curr_metadata = pd.DataFrame({
            'identifier' : identifier, 'group_name' : group_name,
            #'id':self._ids,
            'tomo_path' : self._pickle_path,
            'particle_dir' : self.particle_dir},
                            
            #'left_corner_x' : self._left_corners_rel[:,0],
            #'left_corner_y' : self._left_corners_rel[:,1],
            #'left_corner_z' : self._left_corners_rel[:,2]}
            index = [0]                     
        )

        # deal with self._particle_paths not set
        try:
            self._particle_paths
        except AttributeError:
            self._particle_paths = None
            
        # make current table
        df_dict = {
            'identifier' : identifier, 'group_name' : group_name,
            'id':self._ids,
            'tomo_path' : self._pickle_path,
            'particle_path' : self._particle_paths,
            'left_corner_x' : self._left_corners_rel[:,0],
            'left_corner_y' : self._left_corners_rel[:,1],
            'left_corner_z' : self._left_corners_rel[:,2]}
        if self._boundary_ids is not None:
            b_ids = [tuple(x) for x in self._boundary_ids]
            df_dict['boundary_id'] = b_ids       
        curr_data = pd.DataFrame(df_dict)
    
        # add current metadata
        try:
            self.metadata = self.metadata.append(
                curr_metadata, ignore_index=True)
        except AttributeError:
            self.metadata = curr_metadata
        self.metadata.drop_duplicates().reset_index(drop=True)

        # add current data
        try:
            self.data = self.data.append(curr_data, ignore_index=True)
        except AttributeError:
            self.data = curr_data
