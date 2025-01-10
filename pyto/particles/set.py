"""
Contains class Set for manipulation of particle sets

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
import warnings

import numpy as np
import scipy as sp
try:
    import pandas as pd
except ImportError:
    pass # Python 2
import pyto

class Set(object):
    """
    Contains methods to extract and save particles, and the data about
    the extracted particles.

    Usage:

      particles, labels = Set.extract_particles(...)

    Currently, all instance methods are used for Set.extract_particles()

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
            self, struct=None, work_path=None, box_size=None, particle_dir=None,
            dir_mode=0o775, dtype=None, pixelsize=None, mean=None, std=None):
        """
        Saves specified arguments as attributes of the same name.

        Also sets variables thet describe the standard file organization needed
        to locate tomogram from which particles are extracted.

        Arguments:
          - struct: structure specific (Groups) object that contains the
          structures of interest, such as tethers or connectors
          - work_path: absolute path to the analysis work file
          - segment_var: name of the variable in the catalog files that
          contains the path to the individual dataset pickles corresponding
          to the structure specific object (segments) defined above (arg struct)
          - boundary_var: name of the variable in the catalog files that
          contains the path to the individual dataset pickles corresponding
          to the boundaries object for segments
          - box_size: length of particle box in pixels
          - particle_dir: directory where particles are written
          - dir_mode: mode of the created particle directories
          - dtype: particle dtype
          - pixelsize: pixel size in nm
          - mean: particle means are normalized to this value before saving
          - std: particle stds are normalized to this value before saving
        """

        # set attributes from arguments
        self.struct = struct
        self.work_path = work_path
        #self.segment_var = segment_var
        #self.boundary_var = boundary_var
        self.box_size = box_size
        self.particle_dir = particle_dir
        self.dtype = dtype
        self.dir_mode = dir_mode
        self.pixelsize = pixelsize
        self.mean = mean
        self.std = std

        # data
        self.data = None

        #
        # Standard file organization needed to locate tomogram

        # relative directory of the tomo info file in respect to individual
        # dataset pickles
        self.tomo_info_dir_relative = '../common'

        # tomo info file base (without directory)
        self.tomo_info_base = 'tomo_info'

        # name of the variable in a tomo_info file that contains the name
        # of the tomo file
        self.tomo_path_var = 'image_file_name'


    ##############################################################
    #
    # Extracting particles from multiple datasets
    #
    @classmethod
    def extract_particles(
            cls, struct, work, work_path, group_names, segment_var, 
            identifiers=None, boundary_var=None, 
            box_size=None, particle_dir=None,
            dir_mode=0o775, dtype=None, mean=None, std=None,
            pixelsize=None, label_dtype='int16', label_fg_value=2,
            label_bkg_value=0, bound_fg_value=1, write=True):
        """
        Extracts and writes particles (greyscale subtomos) and the 
        corresponding labels (subtomos of segmented and/or boundary tomos)
        from one or more tomograms. 

        Arg identifiers specify tomograms from which particles are extracted,
        while arg group_names list all experimental groups in which
        contain the tomograms specified by identifiers, Identifiers should
        not be repeated in different experimental groups. If identifiers
        is None, all identifiers of that group are used.

        Particles are defined by property ids of the specified structure
        specific pickle (arg struct). This pickle has to contain data for
        all experiments specified by arg indentifiers. Each experiment
        contains its own ids.

        Requires the standard arrangement of files (see get_tomo_path()).
        Analysis work module, catalog files, structure specific pickles,
        tomograms and individual dataset pickles have to be consistent
        with each other, like in the standard presynaptic analysis.

        In short, the (greyscale) tomogram, labels image and boundary
        image are found in the following way:
          - Catalog variables are read from the analysis work module
          for all experiments
          - Args segment_var and boundary_var specify variable names
          defined in catalogs, these variables contain paths to individual
          tomogram segmentation and boundary pickles.

        For each particle, its positions is determined so that its center
        is the center of mass of the corresponding label. In case the box
        size and the position are such that the particle extends outside the
        tomogram, the particle is shifted as little as needed to fit it
        within the tomogram.

        Tomo and label (output) particle file names contain their
        identifier and id and are saves in the same directory. In addition,
        label particle file names have suffix '_label' and labels+boundaries
        '_label_bound'. 

        Arguments:
          - struct: structure specific Groups object that contains the 
          structures of interest, such as tethers or connectors
          - work: analysis module (needs to have catalog and catalog_directory
          attributes)
          - work_path: path to the analysis work module
          - group_names: dataset (experiment) group names
          - identifiers: dataset (experiment) identifiers
          - segment_var: name of the catalog variable containing path to
          one of the segmentation pickles (such as that defining tethers
          or connectors)
          - boundary_var: name of the catalog variable containing path to
          the boundary pickle (the one that defines vesicles)
          - box_size: particle box size in pixels
          - particle_dir: directory where particles are written is obtained 
          by appending the group name to this variable
          - dir_mode: mode of the created particle directories
          - dtype: tomo particles data type
          - pixelsize: pixel size in nm
          - mean: particle means are set to this value
          - std: particle stds are set to this value
          - label_dtype: label particles data type
          - label_fg_value: forground value for label particles
          - label_bkg_value: background value for label particles
          - write: Flag indicating if particles are written, if False
          everything alse is done, including returning proper tables,
          except writting

        Returns (particles, labels, bounds):
          - particles: Instance of this class containing the tomo data
          - labels: (LabelSet) Instance containing extracted labels data
          - bounds: (BoundarySet) labeled boundaries, returned if 
          boundary_var is soecified
        """

        # make set object for tomo particles
        particles = cls(
            struct=struct, work_path=work_path, box_size=box_size,
            particle_dir=particle_dir, dtype=dtype, mean=mean, std=std,
            pixelsize=pixelsize)

        # make set object for label particles
        from .label_set import LabelSet
        labels = LabelSet(
            struct=struct, work_path=work_path, catalog_var=segment_var,
            box_size=box_size, particle_dir=particle_dir,
            dtype=label_dtype, pixelsize=pixelsize, fg_value=label_fg_value,
            bkg_value=label_bkg_value)

        # make set object for boundaries
        if boundary_var is not None:
            bounds = pyto.particles.BoundarySet(
                struct=struct, work_path=work_path, catalog_var=boundary_var,
                box_size=box_size, particle_dir=particle_dir,
                dtype=label_dtype, fg_value=bound_fg_value,
                pixelsize=pixelsize, bkg_value=label_bkg_value)
        else:
            bounds = None

        # set identifiers if not specified
        if identifiers is None:
            identifiers = np.hstack(
                [particles.struct[g_name].identifiers 
                 for g_name in group_names])

        # loop over specified tomos
        for ident in identifiers:
            group_found = False
            for g_name in group_names:
                if ident not in particles.struct[g_name].identifiers:
                    continue

                # get particle center coordinates (abs and rel) from labels
                labels.get_coordinates_single(
                    work=work, group_name=g_name, identifier=ident)
                if write:
                    labels.particle_dir = os.path.join(particle_dir, g_name) 

                # adjust particle center and left corner coords to the tomo
                particles.get_coordinates_single(
                    work=work, group_name=g_name, identifier=ident,
                    label_set=labels)
                if write:
                    particles.particle_dir = os.path.join(particle_dir, g_name)

                # convert adjusted particle left corner coords to label coords
                labels.set_box_coords(left_corners=particles._left_corners)
                # just for consistency
                labels.set_box_coords(centers=particles._centers)

                # convert adjusted particle left corner coords to 
                # boundary coords
                if boundary_var is not None:
                    bounds.read_tomo(
                        work=work, group_name=g_name, identifier=ident,
                        label_set=labels)
                    bounds.set_box_coords(left_corners=particles._left_corners)
                    # just for consistency
                    bounds.set_box_coords(centers=particles._centers)

                # write particles (needs left corners)
                if write:
                    if particles.pixelsize is not None:
                        pixelsize = particles.pixelsize
                    else:
                        pixelsize = particles._tomo.pixelsize
                    particles.write_particles(
                         identifier=ident, pixelsize=pixelsize, write=write)
                    labels.write_particles(
                        identifier=ident, boundary=bounds, group_name=g_name,
                        pixelsize=pixelsize, write=write)

                # update data
                particles.add_data(group_name=g_name, identifier=ident)
                labels.add_data(group_name=g_name, identifier=ident)

                group_found = True
                break

            # check if the current experiment was processed
            if not group_found:
                raise ValueError(
                    ("Identifier {} does not belong to any of the "
                     + " groups {}").format(ident, group_names))

        if boundary_var is not None:
            return particles, labels, bounds
        else:
            return particles, labels

    @classmethod
    def extract_boundaries(
            cls, bound_struct, label_struct, work, work_path, group_names, 
            segment_var, boundary_var, identifiers=None, box_size=None, 
            particle_dir=None, dir_mode=0o775, dtype=None, mean=None, 
            std=None, pixelsize=None, label_dtype='int16', label_fg_value=2,
            label_bkg_value=0, bound_fg_value=1, write=True):
        """
        Extract (and writes) boundaries from one or more tomograms. It also
        writes particles (greyscale subtomos at the same position) and
        boundaries together with labels obtained by segmentation of these 
        tomograms (such as connectors and tethers).

        Arg identifiers specify tomograms from which particles are extracted,
        while arg group_names list all experimental groups in which
        contain the tomograms specified by identifiers, Identifiers should
        not be repeated in different experimental groups. If identifiers
        is None, all identifiers of that group are used.

        Boundaries are defined by property ids of the specified boundary 
        structure specific pickle (arg bound_struct), while labels are 
        defined by arg label_struct. These pickles have to contain data for
        all experiments specified by arg indentifiers. Each experiment
        contains its own ids.

        Requires the standard arrangement of files (see get_tomo_path()).
        Analysis work module, catalog files, structure specific pickles,
        tomograms and individual dataset pickles have to be consistent
        with each other, like in the standard presynaptic analysis.

        In short, the (greyscale) tomogram, labels image and boundary
        image are found in the following way:
          - Catalog variables are read from the analysis work module
          for all experiments
          - Args segment_var and boundary_var specify variable names
          defined in catalogs, these variables contain paths to individual
          tomogram segmentation and boundary pickles.

        For each particle, its positions is determined so that its center
        is the center of mass of the corresponding label. In case the box
        size and the position are such that the particle extends outside the
        tomogram, the particle is shifted as little as needed to fit it
        within the tomogram.

        Tomo, boundary and boundary+label (output) particle file names 
        contain their identifier and id and are saves in the same directory. 
        In addition, boundary particles have suffix '_bound' and 
        boundary+label '_bound_label'.

        Arguments:
          - bound_struct: structure specific Groups object that contains 
          the boundaries of interest, such as vesicles
          - label_struct: structure specific Groups object that contains the 
          structures of interest, such as tethers or connectors
          - work: analysis module (needs to have catalog and catalog_directory
          attributes)
          - work_path: path to the analysis work module
          - group_names: dataset (experiment) group names
          - identifiers: dataset (experiment) identifiers
          - segment_var: name of the catalog variable containing path to
          one of the segmentation pickles (such as that defining tethers
          or connectors)
          - boundary_var: name of the catalog variable containing path to
          the boundary pickle (the one that defines vesicles)
          - box_size: particle box size in pixels
          - particle_dir: directory where particles are written is obtained 
          by appending the group name to this variable
          - dir_mode: mode of the created particle directories
          - dtype: tomo particles data type
          - pixelsize: pixel size in nm
          - mean: particle means are set to this value
          - std: particle stds are set to this value
          - label_dtype: label particles data type
          - label_fg_value: forground value for label particles
          - label_bkg_value: background value for label particles
          - write: Flag indicating if particles are written, if False
          everything alse is done, including returning proper tables,
          except writting

        Returns (particles, labels, bounds):
          - particles: Instance of this class containing the tomo data
          - labels: (LabelSet) Instance containing extracted labels data
          - bounds: (BoundarySet) labeled boundaries
        """

        # make set object for tomo particles
        particles = cls(
            struct=bound_struct, work_path=work_path, box_size=box_size,
            particle_dir=particle_dir, dtype=dtype, mean=mean, std=std,
            pixelsize=pixelsize)

        # make set object for boundaries
        bounds = pyto.particles.BoundarySet(
            struct=bound_struct, work_path=work_path, catalog_var=boundary_var,
            box_size=box_size, particle_dir=particle_dir,
            dtype=label_dtype, fg_value=bound_fg_value,
            pixelsize=pixelsize, bkg_value=label_bkg_value)

        # make set object for label particles
        from .label_set import LabelSet
        labels = LabelSet(
            struct=label_struct, work_path=work_path, catalog_var=segment_var,
            box_size=box_size, particle_dir=particle_dir,
            dtype=label_dtype, pixelsize=pixelsize, fg_value=label_fg_value,
            bkg_value=label_bkg_value)

        # set identifiers if not specified
        if identifiers is None:
            identifiers = np.hstack(
                [particles.struct[g_name].identifiers 
                 for g_name in group_names])

        # loop over specified tomos
        for ident in identifiers:
            group_found = False
            for g_name in group_names:
                if ident not in particles.struct[g_name].identifiers:
                    continue

                # get particle center coordinates (abs and rel) from bounds
                bounds.get_coordinates_single(
                    work=work, group_name=g_name, identifier=ident)
                if write:
                    bounds.particle_dir = os.path.join(particle_dir, g_name)

                # adjust particle center and left corner coords to the tomo
                particles.get_coordinates_single(
                    work=work, group_name=g_name, identifier=ident,
                    label_set=bounds)
                if write:
                    particles.particle_dir = os.path.join(particle_dir, g_name)

                # convert adjusted particle left corner coords to bound coords
                bounds.set_box_coords(left_corners=particles._left_corners)
                # just for consistency
                bounds.set_box_coords(centers=particles._centers)

                # convert adjusted particle left corner coords to label coords
                labels.get_coordinates_single(
                    work=work, group_name=g_name, identifier=ident)
                labels.set_box_coords(left_corners=particles._left_corners)
                # just for consistency
                labels.set_box_coords(centers=particles._centers)

                # write particles (needs left corners)
                if write:
                    if particles.pixelsize is not None:
                        pixelsize = particles.pixelsize
                    else:
                        pixelsize = particles._tomo.pixelsize
                    particles.write_particles(
                         identifier=ident, pixelsize=pixelsize, write=write)
                    bounds.write_particles(
                        identifier=ident, labels=labels, group_name=g_name,
                        pixelsize=pixelsize, write=write)

                # update data
                particles.add_data(
                    group_name=g_name, identifier=ident)
                bounds.add_data(group_name=g_name, identifier=ident)

                group_found = True
                break

            # check if the current experiment was processed
            if not group_found:
                raise ValueError(
                    ("Identifier {} does not belong to any of the "
                     + " groups {}").format(ident, group_names))

        return particles, labels, bounds


    ##############################################################
    #
    # Extracting particles from one dataset
    #

    def get_coordinates_single(
            self, work, group_name, identifier, label_set):
        """
        Takes particle coordinates from arg label_set and adjusts them
        so that the boxes do not stick out of the greyscale tomogram.

        Arguments:
          - work: analysis module (needs to have catalog and catalog_directory
          attributes)
          - group_name: dataset (experiment) group name
          - identifier: dataset (experiment) identifier
          - pickle_path: path to one of the pickles
          - label_set: LabelSet object corresponding to the current dataset

        Sets attributes:
          - _tomo_path:
          - _tomo:
          - _ids:
          - self._left_corners, self._centers: (ndarry of shape
          N_particles, n_dims) coordinates of left_corners or centers
          of particle boxes
        """

        # get tomo path
        self._tomo_path = self.get_tomo_path(pickle_path=label_set._pickle_path)

        # read tomo
        self._tomo = pyto.grey.Image.read(
            file=self._tomo_path, header=True, memmap=True)

        # get ids
        self._ids = self.struct[group_name].getValue(
            name='ids', identifier=identifier)

        # get coordinates
        self.adjust_box_coords(centers=label_set._centers_abs)


    def get_tomo_info_path(self, pickle_path):
        """
        Depreciated: Moved to set_path.py

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
        Depreciated: Moved to set_path.py

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
        Depreciated: Moved to set_path.py

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
          - _tomo_info_path: tomo info path
          - _tomo_info: tomo info module

        Argument:
          - pickle_path: absolute path to an individual dataset pickle file

        Returns: normalized path to the tomogram
        """

        # try to find tomo_info file
        self._tomo_info_path = self.get_tomo_info_path(pickle_path=pickle_path)

        # import tomo info file
        self._tomo_info = self.import_tomo_info(
            tomo_info_path=self._tomo_info_path)

        # get tomo path
        tomo_path_from_info = self._tomo_info.__getattribute__(
            self.tomo_path_var)
        if os.path.isabs(tomo_path_from_info):
            tomo_path = tomo_path_from_info
        else:
            tomo_info_dir = os.path.split(self._tomo_info_path)[0]
            tomo_path = os.path.join(tomo_info_dir, tomo_path_from_info)

        return os.path.normpath(tomo_path)

    def adjust_box_coords(self, centers):
        """
        Adjusts box coordinates so that they do not stick out of the tomo

        Requires self.box_size

        Argument:
          - centers: box center coordinates, absolute (no inset)

        Sets attributes:
          - self._left_corners
          - self._centers
        """

        # sanity check
        shape = np.array(self._tomo.data.shape)
        if (shape < self.box_size).any():
            raise ValueError(
                "Box size {} is larger than the labels shape {}".format(
                    self.box_size, shape))

        if len(centers) > 0:

            # get left tomo corners from centers and shift if needed
            half_size = self.box_size // 2
            left_coords = centers - half_size
            left_shift = (left_coords < 0)
            left_fixed = np.where(left_shift, 0, left_coords)

            # get right tomo corners from the left and shift if needed
            right_coords = left_fixed + self.box_size
            right_shift = (right_coords >= shape)
            right_fixed = np.where(right_shift, shape, right_coords)

            # set left corners and centers from the (shifted) right corners
            self._left_corners = right_fixed - self.box_size
            self._centers = self._left_corners + half_size

        else:
            self._left_corners = []
            self._centers = []

    def write_particles(
            self, identifier, ids=None, keep_ids=True, pixelsize=None,
            write=True):
        """
        Writes particles as mrc files.

        Particles are normalized to have their mean and std to values
        of self.mean and self.std, respectively, unless they are None.

        Particle file names are formed as follows:
            identifier + "_id-" + id + ".mrc"
        where identifier is given by the argument. If keep_ids is True,
        the existing particle ids are used, otherwise particles are labeled
        from 0 up.

        Perticles are extracted from _tomo attribute. The coordinates
        are determined from _left_corners attribute, _centers are 
        not used.

        Sets attributes:
          - _particle_paths: particle file paths

        Arguments:
          - identifier: dataset identifier
          - ids: particle ids or None to use self._ids
          - keep_ids: flag indicating whether the existing particle are
          used to form the file names
          - pixelsize: pixel size in nm
          - write: if False, particles are not written
        """

        # check ids
        if ids is None:
            ids = self._ids
        if len(ids) == 0:
            self._particle_paths = []
            return

        # make slices
        slices = [
            (slice(coords[0], coords[0] + self.box_size),
             slice(coords[1], coords[1] + self.box_size),
             slice(coords[2], coords[2] + self.box_size))
            for coords in self._left_corners]

        # make particle paths
        if keep_ids:
            n_digits = np.log10(ids.max()).astype(int) + 1
            particle_names = [
                ('{}_id-{:0' + str(n_digits) + 'd}.mrc').format(
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
        for sl, path in zip(slices, self._particle_paths):

            # get particle data
            particle_data = self._tomo.useInset(
                inset=sl,  mode=u'absolute', expand=False, update=False)

            # put code to process particle here
            if self.std is not None:
                particle_data = self.std * particle_data / particle_data.std()
            if self.mean is not None:
                particle_data = particle_data - particle_data.mean() + self.mean

            # clip, set to dtype
            if self.dtype is not None:
                try:
                    range_info = np.finfo(self.dtype)
                except ValueError:
                    range_info = np.iinfo(self.dtype)
                if ((particle_data < range_info.min).any() or
                    (particle_data > range_info.max).any()):
                    warnings.warn(
                        ("Warning: Particle {} needs to be clipped to convert"
                         + " it to {}".format(path, self.dtype)))
                particle_data = np.clip(
                    particle_data, range_info.min, range_info.max)
                particle_data = particle_data.astype(
                    self.dtype, casting='unsafe')

            # write
            particle = pyto.grey.Image(data=particle_data)
            if write:
                particle.write(
                    file=path, header=self._tomo.header, pixel=pixelsize)

    def add_data(self, group_name, identifier):
        """
        Adds current data to self.metadata and self.data pandas tables.

        Doesn't do anything if this object has no ids (self._ids).

        Arguments:
          - group_name: dataset (experiment) group name
          - identifier: dataset (experiment) identifier
        """

        if len(self._ids) == 0: return

        # make current meta table
        curr_metadata = pd.DataFrame({
            'identifier' : identifier, 'group_name' : group_name,
            #'id':self._ids,
            'tomo_path' : self._tomo_path,
            'particle_dir' : self.particle_dir},
            #'left_corner_x' : self._left_corners[:,0],
            #'left_corner_y' : self._left_corners[:,1],
            #'left_corner_z' : self._left_corners[:,2]
            index = [0]
        )

        # deal with self._particle_paths not set
        try:
            self._particle_paths
        except AttributeError:
            self._particle_paths = None
            
        # make current table
        curr_data = pd.DataFrame({
            'identifier' : identifier, 'group_name' : group_name,
            'id':self._ids,
            'tomo_path' : self._tomo_path,
            'particle_path' : self._particle_paths,
            'left_corner_x' : self._left_corners[:,0],
            'left_corner_y' : self._left_corners[:,1],
            'left_corner_z' : self._left_corners[:,2]}
        )

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
