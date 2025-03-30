"""
Contains class MultiParticleSets that holds particle coordinates from  
multiple classifications

# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"


import os
import re
import pickle
from copy import copy, deepcopy

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
import pandas as pd

import pyto
from ..io.pandas_io import PandasIO
from pyto.segmentation.labels import Labels
from pyto.particles.relion_tools import get_array_data
import pyto.spatial.coloc_functions as col_func
from .point_pattern import get_region_coords, exclude, project
from .particle_sets import ParticleSets 
from .line_projection import LineProjection
from .mps_conversion import MPSConversion
from .mps_interconversion import MPSInterconversion
from .mps_analysis import MPSAnalysis


class MultiParticleSets(MPSConversion, MPSInterconversion, MPSAnalysis):
    """
    Read and organize particle sets from multiple regions and classifications.

    Data attributes:
      - tomos: (pandas.DataFrame) tomos, each row specifies one tomo
      - particles: (pandas.DataFrame) particles, each row specifies one 
      particle

    Methods that convert instance of this class from / to other formats:
      - make(mode='pyseg'), or make_pyseg(): converts pyseg-format
      tomo and particle star files to an instance of this class
      - from_star_picks: converts pyseg/relion particle picks star file
      to an instance of this class
      - from_particle_sets(), to_particle_sets(): converts data from / to 
      spatial.ParticleSets
      - from_coords(): converts a dataframe containing particles to
      instance of this class
      - from_mps(): reads multiple particles sets previously saved as
      individual instances of this class and combines them in one 
      instance of this class, so that it can be used for colocalization
      (ColocLite.colocalize())

    I/O:
      - write(), read(): write and read an instance of this class

    Particle manipulation methods:
      - convert(): converts particle coordinates from one (tomo) frame to
      another and project them on a rwgion
      - select(): selects subset of particles from the current instance
      - exclude(): impose exclusion distance
      - find_inside(): find particles that are inside a given image shape
      - extracts_combination_sets_n(): extracts particles of combination
      sets (those composed of multiple class numbers)
      - find_min_distances(): find closest points (particles) between two
      sets
    """

    def __init__(self):
        """Sets default values of attributes.

        """

        # star file labels 
        self.micrograph_label = 'rlnMicrographName'
        self.seg_image_label = 'psSegImage'
        self.string_pattern = ['Name', 'Image']
        self.image_label = 'rlnImageName'
        self.ctf_label = 'rlnCtfImage'
        self.region_origin_labels = ['psSegOffX', 'psSegOffY', 'psSegOffZ']
        self.rotation_labels = ['psSegRot', 'psSegTilt', 'psSegPsi']
        self.coord_labels = [
            'rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']
        self.particle_rotation_labels = [
            'rlnAngleTilt', 'rlnAngleTiltPrior', 'rlnAnglePsi',
            'rlnAnglePsiPrior', 'rlnAngleRot']
        self.origin_labels = ['rlnOriginX', 'rlnOriginY', 'rlnOriginZ']
        self.angst_labels = [
            'rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']
        self.class_label = 'rlnClassNumber'
        self.particle_star_label = 'psStarFile'
       
        # dataframe columns
        self.tomo_id_col = 'tomo_id'
        self.coord_bin_col = 'coord_bin'
        self.region_col = 'region'
        self.region_id_col = 'region_id'
        self.region_bin_col = 'region_bin'
        self.region_offset_cols = [
            'region_offset_x', 'region_offset_y', 'region_offset_z'] 
        self.region_shape_cols = [
            'region_shape_x', 'region_shape_y', 'region_shape_z'] 
        self.pixel_nm_col = 'pixel_nm'
        self.exclusion_col = 'exclusion_nm'
        self.particle_id_col = 'particle_id'

        # original coords in the initial frame (before projection) 
        self.orig_coord_cols = ['x_orig', 'y_orig', 'z_orig']

        # original coordinates in the region frame (before projection) 
        self.orig_coord_reg_frame_cols = [
            'x_orig_reg_frame', 'y_orig_reg_frame', 'z_orig_reg_frame']

        # projected coords in the region frame
        self.coord_reg_frame_cols = [
            'x_reg_frame', 'y_reg_frame','z_reg_frame' ]
        
        # projected coords converted back to the initial frame
        self.coord_init_frame_cols = [
            'x_init_frame', 'y_init_frame','z_init_frame' ]

        self.coord_cols = ['coord_x', 'coord_y', 'coord_z']
        self.class_name_col = 'class_name'
        self.class_number_col = 'class_number'
        self.subclass_col = 'subclass_name'

        self.pick_index_col = 'pick_index'

        # whether to keep after exclusion
        self.keep_col = 'keep'
        
        # whether original coords inside region (segment)
        self.in_region_col = 'in_region'

        # pickled table suffices
        self.pickle_tomos_suffix = '_tomos'
        self.pickle_particles_suffix = '_particles'
        
    #
    # IO
    #

    def __getstate__(self):
        """To avoid pickling dataframes.
        """
        state = self.__dict__.copy()
        # Don't pickle tomos and particles
        del state["tomos"]
        del state['particles']
        return state

    def __setstate__(self, state):
        """Just to complement __getstate__(), probably not needed.
        """
        self.__dict__.update(state)
        
    def write(self, path, verbose=True, out_desc="", info_fd=None):
        """Writes this instance in the form that can be read by read().

        Pickles this instance without self.tomos and self.particles dataframes. 

        Tomos and particles tables are writen separately using
        ..io.pandas_io.write(). The file paths are obtained by adding 
        subscripts self.pickle_tomos_suffix and self.pickle_particles_suffix,
        respectively, to arg path (see ..io.pandas_io.write() docs for more
        info about the write formats.

        Arguments:
          - path: common part of the file path expected to have 
          extension 'pkl'
          - verbose: flag indicating if a statement is printed for every
          file that is written
          - out_desc: description of the file that is written, used
          only for the out messages if self.verbose is True 
          - info_fd: file descriptor for the above write messages

        """

        # make directory if needed
        dir_ = os.path.split(path)[0]
        if len(dir_) > 0:
            os.makedirs(dir_, exist_ok=True)
        
        # save tomos and particles
        split_path = path.rsplit('.', maxsplit=1)
        tomos_path = (
            f"{split_path[0]}{self.pickle_tomos_suffix}.{split_path[1]}")
        PandasIO.write(
            table=self.tomos, base=tomos_path, file_formats=['pkl', 'json'],
            verbose=False, info_fd=info_fd)
        parts_path = (
            f"{split_path[0]}{self.pickle_particles_suffix}.{split_path[1]}")
        PandasIO.write(
            table=self.particles, base=parts_path,
            file_formats=['pkl', 'json'], verbose=False, info_fd=info_fd)

        # pickle the rest (see __getstate__)
        with open(path, 'wb') as fd:
            pickle.dump(self, fd)
            if verbose:
                print(f"Pickled {out_desc} MPS object to {path}", file=info_fd)

    @classmethod
    def read(cls, path, verbose=True, out_desc="", info_fd=None):
        """Reads instance of this class from files created by write().

        Reads pickled instance of this class that does not contain self.tomos
        and self.particles dataframes and saved versions of these
        dataframes from (three) separate files. Uses these to make an 
        instance of this class.

        Arguments:
          - path: common part of the file path
          - verbose: flag indicating if a statement is printed for ever
          file that is read
          - out_desc: description of the file that is read, used
          only for the out messages if self.verbose is True 
          - info_fd: file descriptor for the above read messages
       Returns instance of this class.
        """
        
        # read the rest
        with open(path, 'rb') as fd:
            inst = pickle.load(fd)

        # read tomos and particles
        split_path = path.rsplit('.', maxsplit=1)
        tomos_path = (
            f"{split_path[0]}{inst.pickle_tomos_suffix}.{split_path[1]}")
        inst.tomos = PandasIO.read(
            base=tomos_path, file_formats=['pkl', 'json'], verbose=False,
            info_fd=info_fd)
        parts_path = (
            f"{split_path[0]}{inst.pickle_particles_suffix}.{split_path[1]}")
        inst.particles = PandasIO.read(
            base=parts_path, file_formats=['pkl', 'json'], verbose=False,
            info_fd=info_fd)
        
        if verbose:
            print(f"Read {out_desc} MPS object {path}", file=info_fd)

        return inst

    def copy(self):
        """Returns a (deep) copy of this instance.

        Dataframes (self.tomos and self.particles) are copied separatelly
        and added as attributes to th resulting instance.
        """

        mps = deepcopy(self)
        mps.tomos = self.tomos.copy()
        mps.particles = self.particles.copy()
        return mps
    
    #########################################################
    #
    # Low-level methods, generally used in methods that make / convert
    # instances of this class
    #
        
    def read_star(
            self, path, mode, tomo_ids=None, tomo_id_mode='munc13',
            tomo_id_func=None, tomo_id_kwargs={},
            pixel_size_nm=None, coord_bin=1., tablename='data_',
            do_origin=True, find_region_shape=True,
            class_name='', class_number=None,
            tomos=None, keep_star_coords=True, keep_all_cols=False):
        """Makes particles or tomos table from star files.

        Arg mode determines whether tomos or particles table is made.

        Tomo ids are determined in both modes from tomo paths (column 
        self.micrograph_label) in the following order:
          - If tomo_id_mode is specified, 
            coloc_functions.get_tomo_id(tomo_path, tomo_id_mode) is used
          - If tomo_id_func is specified, 
            tomo_id_func(tomo_path, **tomo_id_kwargs)
          - Otherwise, the tomo paths are copied to tomo ids

        In tomo mode make the following columns:
          - self.tomo_id_col
          - column for args pixel_size and coord_bin
          - determines shape of region images (if find_region_shape = True)

        In particle mode, does the following: 
          - Converts star column self.micrograph_label (default 
          'rlnMicrographName') to tomo ids and saves them to 
          self.tomo_id_col (default 'tomo_id') column (conversion
          mode specified as arg tomo_id_mode)
          - Attempts to extract particle id from self.image_label column of 
          self.particles and to put it in column self.particle_id_col. If 
          column self.image_label doesn't exist, sets -1 in 
          self.particle_id_col (default 'particle_id').
          - Calculates particle coordinates from star file coordinate 
          and offset columns ('rlnCoordinateXYZ', 'rlnOriginXYZ' and
          'rlnOriginXYZAngst', see self.get_original_coords() for more info)
          and saves them in self.orig_coord_cols columns (defailt
          'x_orig', 'y_orig', 'z_orig') 
          - Column 'class_name' contans the value of arg class_name
          - Saves star column self.class_label (default 'rlnClassNumber') 
          as self.particles column 'class_number' if arg class_number
          is None, otherwise ise the value of arg class_number
          - Saves star columns 'rlnCoordinateX', 'rlnCoordinateY',
          'rlnCoordinateZ', 'rlnAngleTilt', 'rlnAngleTiltPrior', 
          'rlnAnglePsi', 'rlnAnglePsiPrior', 'rlnAngleRot', 'rlnOriginX', 
          'rlnOriginY', 'rlnOriginZ' as columns of self.particles with same
          names

        Arguments:
          - path: tomos or particles star file path
          - mode: 'tomo' or 'particle', for tomo or particle star files, 
          respectively
          - tomo_ids: ids of tomo that should be considered, None for
          all tomos (default)
          - tomo_id_mode: mode for determining tomo id, passed directly to 
          coloc_functions.get_tomo_id(mode)
          - tomo_id_func: function that extracts tomo ids from paths, 
          the first argument has to be tomo path
          - tomo_id_kwargs: kwargs for tomo_id_func
          - pixel_size_nm: (single number, dict where keys are tomo ids, or 
          pandas.DataFrame having tomo_id column)
          pixel size [nm] of the system in which particle coordinats are given, 
          needed in mode 'tomo'
          - coord_bin: (single number, dict where keys are tomo ids, or 
          pandas.DataFrame having tomo_id column) binning
          factor of the system in which particle coordinats are given,
          needed in mode 'tomo'
          - tablename: name of the table that contains data (default "data_")
          - do_origin: flag indicating if coordinates are adjusted for
          origin shifts (rlnOriginX/Y/Z and rlnOriginX/Y/ZAngst)
          - find_region_shape: flag indicating if shape of regions is
          determined (by reading region image data)
          - class_name: name of the classification, needed in mode 'particle'  
          - class_number: class number, needed in mode 'particle'  
          - tomos: tomo star file, needed in mode 'particle' when 
          _rlnOriginX/Y/ZAngst labels are present, has to
          have pixel size (nm) in column self.pixel_nm_col 
          - keep_star_coords: If True, star file coordinates and offsets
          (columns rlnCoordinateX/Y/Z and rlnOriginX/Y/Z or 
          rlnOriginX/Y/ZAngst are saved in the resulting particles table
          - keep_all_cols: if True, all columns of the star file are 
          kept (default False)
 
        Sets attributes:
          - self.tomo_cols_all: all tomo column names, contain all 
          star file fields and the calculated columns (mode == 'tomo')
          - self.tomo_cols: column names of the returned tomo table
          (mode == 'tomo')
          - self.particle_cols_all: all particle column names, contain all 
          star file fields and the calculated columns (mode == 'particle')
          - sels.particle_cols: column names of the returned particle table
          (mode == 'particle')

        Returns tomo or particle table
        """

        # read star and convert fields to numbers
        #if isinstance(path, pd.DataFrame):  # better in add_star_pickles
        #    table = path
        #else:
        table = pd.DataFrame(
            get_array_data(starfile=path, tablename=tablename, types=str))
        for col in table.columns:
            if np.any([pat in col for pat in self.string_pattern]):
                table[col] = table[col].astype(str) 
            else:
                try:
                    table[col] = pd.to_numeric(table[col])
                except ValueError:
                    pass
                    
        # add tomo ids
        self.add_tomo_ids(
            table=table, path_col=self.micrograph_label,
            tomo_id_mode=tomo_id_mode,
            tomo_id_func=tomo_id_func, tomo_id_kwargs=tomo_id_kwargs)
        #if tomo_id_mode is not None:
        #    tomo_id_func = col_func.get_tomo_id
        #    tomo_id_kwargs = {'mode': tomo_id_mode}
        #if tomo_id_func is not None:
        #    table[self.tomo_id_col] = table[self.micrograph_label].map(
        #        lambda x: tomo_id_func(x, **tomo_id_kwargs))
        #else:
        #    table[self.tomo_id_col] = table[self.micrograph_label]
            
        # select tomos
        if tomo_ids is not None:
            table = table[table[self.tomo_id_col].apply(
                lambda x: x in tomo_ids)]

        # tomo specific
        if mode == 'tomo':

            # region
            table[self.region_col] = table[self.seg_image_label]

            # pixel size and bin
            table = self.set_column(
                tomos=table, column=self.pixel_nm_col, value=pixel_size_nm,
                update=True)
            table = self.set_column(
                tomos=table, column=self.coord_bin_col, value=coord_bin,
                update=True)

            # rename region offset
            #col_rename = dict([(old, new) for old, new in zip(
            #    self.region_origin_labels, self.region_offset_cols)])
            col_rename = dict(
                zip(self.region_origin_labels, self.region_offset_cols))
            if keep_all_cols:
                for old, new in col_rename.items():
                    table[new] = table[old]
            else:
                table = table.rename(columns=col_rename)

            # rotation columns
            self.tomo_cols_all = table.columns
            #rot_cols = table.columns[5:8].tolist()
            rot_cols = [col for col in table.columns
                        if col in self.rotation_labels]
            
            # list of important columns
            out_cols = (
                [self.tomo_id_col, self.region_col] + self.region_offset_cols
                + [self.pixel_nm_col, self.coord_bin_col] + rot_cols)
            self.tomo_cols = out_cols

            # get region shapes
            if find_region_shape:
                abs_dir = os.path.abspath(os.path.dirname(path))
                self.get_region_shapes(
                    tomos=table, curr_dir=abs_dir, update=True)
            
        # particle star specific: particle ids, calculated coords
        if mode == 'particle':

            # particle id
            try:
                table[self.particle_id_col] = table[self.image_label].map(
                    lambda x: self.get_particle_id(path=x))
            except KeyError:
                if self.image_label not in table.columns:
                    table[self.particle_id_col] = -1
                else:
                    raise
            out_cols = [self.tomo_id_col, self.particle_id_col]

            # copy from tomo
            if tomos is not None:
                table[self.pixel_nm_col] = table[[self.tomo_id_col]].merge(
                    tomos[[self.tomo_id_col, self.pixel_nm_col]],
                    on=self.tomo_id_col, how='left')[self.pixel_nm_col]
                out_cols.append(self.pixel_nm_col)

            # calculate coords
            table = self.get_original_coords(table=table, do_origin=do_origin)
            if keep_star_coords:
                out_cols += self.coord_labels
                rot_cols = [
                    lab for lab in self.particle_rotation_labels
                    if lab in table.columns]
                out_cols += rot_cols
                if self.origin_labels[0] in table.columns:
                    out_cols += self.origin_labels
                if self.angst_labels[0] in table.columns:
                    out_cols += self.angst_labels
            out_cols += self.orig_coord_cols

            # class name and number
            table[self.class_name_col] = class_name
            if class_number is None:
                class_number = table[self.class_label]
            table[self.class_number_col] = class_number
            out_cols += [self.class_name_col, self.class_number_col]
            
            # prepare output
            self.particle_cols_all = table.columns
            self.particle_cols = out_cols

        if not keep_all_cols:
            table = table[out_cols]
        return table

    def add_tomo_ids(
            self, table, path_col=None,
            tomo_id_mode='munc13', tomo_id_func=None, tomo_id_kwargs={}):
        """Generates and adds tomo ids to a table.

        Tomo ids can be extracted from any column of arg tables. Often,
        they are determined from tomo paths. 

        While coloc_functions.get_tomo_id() contains several predefined
        modes to determine tomo ids (see docs there), a custom
        defined function can be specified. 

        Tomo ids are determined from column (arg) path_col of table
        (arg) table, in the following order:
          - If tomo_id_mode is specified, 
            coloc_functions.get_tomo_id(tomo_path, tomo_id_mode) is used
          - If tomo_id_func is specified, 
            tomo_id_func(tomo_path, **tomo_id_kwargs)
          - Otherwise, the tomo paths are copied to tomo ids

        The generated tomo ids are added to column self.tomo_id_col.

        Arguments:
          - table (pandas.DataFrame) table 
          - path_col: column of tables wrom which tomo ids are determined,
          if None (default) self.micrograph_label is used 
          - tomo_id_mode: mode for determining tomo id, passed directly to 
          coloc_functions.get_tomo_id(mode)
          - tomo_id_func: function that extracts tomo ids 
          - tomo_id_kwargs: kwargs for tomo_id_func        

        Modifies table, doesn't return anything
        """
        
        if path_col is None:
            path_col = self.micrograph_label
        if tomo_id_mode is not None:
            tomo_id_func = col_func.get_tomo_id
            tomo_id_kwargs = {'mode': tomo_id_mode}
        if tomo_id_func is not None:
            table[self.tomo_id_col] = table[path_col].map(
                lambda x: tomo_id_func(x, **tomo_id_kwargs))
        else:
            table[self.tomo_id_col] = table[path_col]
    
    def get_particle_id(self, path):
        """Extracts particle id from subtomo path.
        """
        name = os.path.basename(path).split('.')[0]
        pid = int(name.split('_')[-1])
        return pid

    def get_original_coords(self, table, do_origin=True):
        """Calculate coordinates from relion star file columns.

        Coordinates are calculated from star file columns using:
          coords - coords_origin - coords_angst / pixel_size_angst
        where :
          - coords: subtomo coords from fields self.coord_labels 
          ('rlnCoordinateX/Y/Z' in relion) [pixel]
          - coords_origin: alignment corrections from fields 
          self.origin_labels ('rlnOriginX/Y/Z' in relion) [pixel]
          - coords_angst: alignment corrections from fields 
          self.angst_labels ('rlnOriginX/Y/ZAngst' in relion) [A]
        
        The calculated coords are added to (arg) table as new coulumns.

        Returns particle table that contains all data from (arg) tables
        and the calculated coords columns.
        """

        # loop over axes
        for ind in range(len(self.coord_labels)):        
            coord = table[self.coord_labels[ind]].copy()
            corrected = False
            if (do_origin and (self.origin_labels is not None)
                and (self.origin_labels[ind] in table.columns)):
                coord -= table[self.origin_labels[ind]]
                corrected = True
            if (do_origin and (self.angst_labels is not None)
                and (self.angst_labels[ind] in table.columns)):
                coord -= (table[self.angst_labels[ind]]
                          / (10 * table[self.pixel_nm_col]))
                corrected = True
            if corrected:
                table[self.orig_coord_cols[ind]] = coord.round().astype(int)
            else:
                table[self.orig_coord_cols[ind]] = table[
                    self.coord_labels[ind]]
                
        return table
    
    def set_region_offset(self, tomos=None, offset=None, update=False):
        """Figure out region offset.

        Makes pandas.DataFrame that contains tomo ids (self.tomo_id_col)
        and region offset columns (self.region_offset_cols), and possibly
        region shape columns (self.region_shape_cols).

        Offsets can be specified in the following ways:
          - offset=0: no offset for any of the tomos
          - offset=tab: pandas.DataFrame: has to contain tomo_ids 
          (self.tomo_id_col) and offset (self.region_offset_cols) columns
          - offset='path.csv': csv file where column 0 contains tomo ids,
          columns 1-3 offsets and columns 4-6 region shapes
          - offset=None: offsets are expected to be present in tomo columns
          self.region_origin_labels

        If update is True and the input value of arg tomos contains columns
        (self.region_offset_cols, these values are overwritten by the values
        specified by arg offset.
        """

        # figure out offset
        #reg_offsets = None
        if offset is not None:

            # make reg_offsets dataframe
            if isinstance(offset, pd.DataFrame):
                reg_offsets = offset
            elif isinstance(offset, str):  # csv file name
                extension = os.path.splitext(offset)[1]
                if extension == '.csv':
                    reg_offsets = pd.read_csv(offset, sep=" ")
                    reg_offsets.columns = (
                        [self.tomo_id_col] + self.region_offset_cols
                        + self.region_shape_cols)
                else:
                    raise ValueError(
                        f'Offsets file {offset} has to be in csv format.')
            elif isinstance(offset, (int, float)) and (offset == 0):  # 0
                tomo_ids = tomos[self.tomo_id_col]
                offset_vals = np.zeros(shape=(len(tomo_ids), 3))
                reg_offsets = pd.DataFrame({
                    self.tomo_id_col: tomo_ids})
                reg_offsets_val = pd.DataFrame(
                    offset_vals, columns=self.region_offset_cols)
                reg_offsets = reg_offsets.join(reg_offsets_val)

        else:

            # offset in tomos, make reg_offset dataframe 
            reg_offsets = tomos[[self.tomo_id_col] + self.region_origin_labels]
            reg_offsets.columns = (
                [self.tomo_id_col] + self.region_offset_cols)
            
        if update:
            tomo_cols = tomos.columns
            try:
                tomo_cols = tomo_cols.drop(self.region_offset_cols)
            except KeyError:
                pass
            reg_offsets = pd.merge(
                tomos[tomo_cols],
                reg_offsets[[self.tomo_id_col] + self.region_offset_cols],
                on=self.tomo_id_col, how='left', sort=False)
            
        return reg_offsets
                    
    def set_column(self, column, value=None, tomos=None, update=False):
        """Sets a tomogram column to spacified value(s).  
        
        The value can be specified as:
          - a single number that is used for all tomos
          - dict value where keys are tomo ids and values are values
          - pandas.DataFrame table that has tomo ids in column  
          self.tomo_id_col and the values in column named arg column

        First makes pandas.DataFrame that contains only tomo ids 
        (self.tomo_id_col) and the specified column (arg column). If arg 
        update is True, adds this column to the tomo table (arg tomos).

        If arg update is True, it returns a dataframe with the 
        specified column added. That is, it does not add a column
        to the object specified by arg tomos.

        If update is True, arg value is not None and the input value of 
        arg tomos contains a column named (arg) column, these values are 
        overwritten by the values specified by arg value.

        Arguments:
          - column: column name
          - tomos: tomos dataframe (like self.tomos)
          - value: (single number, dict or pandas.DatoFrame) value to be 
          added
          - update: indicates whether the new values are added to the 
          (arg) tomo, or a dataframe with columns self.tomo_id_col and
          (arg) column) are returned
        """
        
        # put column value(s) in a dataframe
        if value is not None:  # looks funny but actually ok
            if isinstance(value, (int, float)):  # single number
                local_tab = pd.DataFrame({
                    self.tomo_id_col: tomos[self.tomo_id_col],
                    column: value}) 
            elif isinstance(value, dict):  # dict
                local_tab = pd.DataFrame({
                    self.tomo_id_col: value.keys(),
                    column: value.values()})
            elif isinstance(value, pd.DataFrame):  # dataframe
                local_tab = value
            else:
                raise ValueError(
                    "Arg value has to be int, float, dict or pandas.DataFrame")

        # column values in tomos
        elif column in tomos.columns: 
            local_tab = tomos[[self.tomo_id_col, column]]

        # no column values
        else:
            local_tab = None
            
        if update:
            if value is not None:
                tomo_cols = tomos.columns
                try:
                    tomo_cols = tomo_cols.drop(column)
                except KeyError:
                    pass
                tomos = pd.merge(
                    tomos[tomo_cols], local_tab[[self.tomo_id_col, column]],
                    on=self.tomo_id_col, how='left', sort=False).set_index(
                        tomos.index)
            return tomos
        else:
            return local_tab
                    
    def set_region_id(self, tomos=None, region_id=None, update=False):
        """Sets region id.

        Makes pandas.DataFrame that contains only tomo ids (self.tomo_id_col)
        and region id columns (self.region_id_col). If arg update is True, adds
        region id to the tomo dable (arg tomos).

        Region id can be specified in the following ways:
          - region_id=n: the same (int) region id for all tomos
          - region_id=tab: pandas.Dataframe: has to contain tomo ids 
          (self.tomo_id_col) and region id (self.region_id_col) columns
          - region_id=None: region ids are expected to be present in tomo
          coulmn self.region_id_col

        """
        tab = self.set_column(
            tomos=tomos, value=region_id, column=self.region_id_col,
            update=update)
        return tab

    def set_region_bin(self, tomos=None, region_bin=None, update=False):
        """Sets region bin.


        """
        tab = self.set_column(
            tomos=tomos, value=region_bin, column=self.region_bin_col,
            update=update)
        return tab

    def get_region_shapes(self, tomos, curr_dir=None, update=False):
        """Finds shape of region tomos.

        Gets region paths from column self.region_col of arg tomos, 
        prepends arg curr_dir if the region paths are relative,
        reads the regions and determines their shape.

        The shapes are saved in columns self.region_shape_cols.

        Arguments:
          - tomos: tomo table, like self.tomos
          - curr_dir: directory in respect to which the region paths are 
          specified, os.getcwd() if None
          - update: flag that specifies wheteher the shapes are added as
          new column to arg tomos 

        Returns:
          - tomos with added shapes if update is True
          - pandas DataFrame that contains tomo ids (self.tomo_ids column)
          and shapes if update is False
        """

        if curr_dir is None:
            curr_dir = os.getcwd() 
        
        # get shapes data
        shapes = []
        region_paths = tomos[self.region_col]
        for reg_path in region_paths:
            if not os.path.isabs(reg_path):
                reg_path = os.path.join(curr_dir, reg_path)
            shape_one = pyto.segmentation.Labels.read(
                file=reg_path, memmap=True).data.shape
            shapes.append(shape_one) 
        shape_np = np.vstack(shapes)

        if update:
            tomos[self.region_shape_cols] = shape_np
            return tomos
        else:
            res = tomos[[self.tomo_id_col]].copy()
            res[self.region_shape_cols] = shape_np
            return res
        
    #
    # Methods that calculate new particle properties
    #

    def convert(
            self, tomos, particles, region_bin=None, region_offset=None,
            region_id=None, project=True, exclusion=None,
            exclusion_mode='before_projection', class_name_col=None,
            remove_outside_region=False, inside_coord_cols=None):
        """Converts coords to the region frame and projects them on region.

        Converts particle coordinates from the initial (original) tomo frame
        to the corresponding region tomo for multiple tomos. Initial and
        egion tomos can have different bin factors and the region tomos can
        have coordinate offsets in respect to the original tomos.

        During the coversion, coordinates are trasformed first according to 
        the bin factors and then to offsets (see convert_one() doc for
        more info.

        Particles may be excluded based on two criteria:
          - If arg exclusion is not None, min distance between neighboring 
          particles (value of arg exclusion) is imposed, that is the minimum
          number of particles are labeled so that the unlabeled particles 
          do not have any nighbor closer than distance given by arg
          exclusion (see exclude_single_doc() for more info). This 
          exclusion may be imposed either before or after 
          particle projection (determined by arg exclusion_mode)
          - If arg remove_outside_region is True, particles that 
          after conversion to regions frame fall outside region image 
          are labeled. 
        Particles that are selected (labeled) by distance or inside region 
        exclusion are kept in the particles table. The labeled (excluded)
        particles are assigned value False in self.keep_col column, while 
        non-excluded particles have True in that column.
        
        If args region_bin and region_offset are None and these are not in
        tomos table, coordinates are not converted to
        the region frame, nor are the projected coords converted back to 
        the original frame. In this case, arg particles is expected to contain
        column self.orig_coord_reg_frame_cols.

        Arguments:
          - tomos: (pandas.DataFrame) tomos table, like self.tomos
          - particles (pandas.DataFrame) particles table, like self.particles
          - region_bin: bin factor of the region images 
          - region offset: offset of region images relative to the initial
          tomo frame, specified as expalined in set_region_offset() doc.
          - region_id: region id in region images
          - project: flag indicating whether coordinates are projected
          on regions after coversion
          - exclusion: exclusion distance (>0, in nm) or None for no 
          distance based exclusion
          - exclusion_mode: distance exclusion is applied on particles
          before the projection (after conversion) if 'before_projection',
          or after the projection if 'after_projection'
          - class_name_col: column that contains particle set names (classes),
          used to separate particles in individual sets for exclusion, 
          if None self.class_name col is used
          - remove_outside_region: flag indicating whether coordinates
          that fall outside region images after conversion are excluded
          - inside_coord_cols: column names of coordinates that are
          tested whether they are inside region image shape, if None 
          set to columns self.orig_coord_reg_frame_cols (particles
          in regions frame before projection to regions)

        Returns: (pandas.DataFrame) updated particles table
        """

        if class_name_col is None:
            class_name_col = self.class_name_col
        
        # add tomo properties
        if region_offset is not None:
            tomos = self.set_region_offset(
                tomos=tomos, offset=region_offset, update=True)
        if region_id is not None:
            tomos = self.set_region_id(
                tomos=tomos, region_id=region_id, update=True)
        if region_bin is not None:
            tomos = self.set_region_bin(
                tomos=tomos, region_bin=region_bin, update=True)
        if exclusion is not None:
            tomos = self.set_column(
                column=self.exclusion_col, tomos=tomos, value=exclusion,
                update=True)
            if exclusion_mode not in ['before_projection', 'after_projection']:
                raise ValueError(
                    f"Argument exclusion_mode ({exclusion_mode}) has to be "
                    + "'before_projection' or 'after_projection'")
            
        # loop over tomos
        particles_found = False
        for _, tomo_row in tomos.iterrows():

            # get tomo data
            tomo_id = tomo_row[self.tomo_id_col]
            try: 
                bin_one = (tomo_row[self.coord_bin_col]
                           / tomo_row[self.region_bin_col])
            except KeyError:
                bin_one = None
            try:
                offsets_one = (tomo_row[self.region_offset_cols]
                               .to_numpy(dtype='int'))
            except KeyError:
                offsets_one = None
            reg_path = tomo_row.get(self.region_col, None)
            reg_id = tomo_row.get(self.region_id_col, None)
            pix_nm = tomo_row.get(self.pixel_nm_col, None)
            try:
                excl_nm = tomo_row[self.exclusion_col]
            except KeyError:
                excl_nm = None
                
            # extract particles for the current tomo
            part_one = particles[particles[self.tomo_id_col]==tomo_id].copy()

            # get regions shape
            if remove_outside_region:
                region_shape = tomo_row[self.region_shape_cols]
            else:
                region_shape = None
                
            # convert coords to regions and project 
            part_current = self.convert_one(
                particles=part_one, bin=bin_one, offsets=offsets_one,
                region_path=reg_path, region_id=reg_id,
                pixel_nm=pix_nm, project=project, exclusion=excl_nm,
                exclusion_mode=exclusion_mode, class_name_col=class_name_col,
                region_shape=region_shape,
                inside_coord_cols=inside_coord_cols)

            # merge with all particles
            if part_current.shape[0] > 0:
                try:
                    particles_final = pd.concat([particles_final, part_current])
                except (AttributeError, UnboundLocalError):
                    particles_final = part_current
                    particles_found = True

            # convex hull

        if particles_found:
            return particles_final
        else:
            return None
            
    def convert_one(
            self, particles, bin=1, offsets=None, region_path=None,
            region_id=None, pixel_nm=1, project=True, exclusion=None,
            exclusion_mode='before_projection', class_name_col=None,
            region_shape=None, inside_coord_cols=None):
        """Converts coords to the region frame and projects them, for one tomo.

        Arg particles is expected to contain particles of one tomo,
        and one or more particle sets (classes). For multiple tomos,
        use convert().

        Particle coordinates in the region frame are obtained as:
            orig_frame_coords * bin - offsets 
        where bin is expected to be specified as:
            bin = original_tomo_bin / regions_tomo_bin

        Particle coordinates (as specified in self.orig_coord_cols column
        of particles table) have to be given in the full tomo at the 
        original tomo bin.

        If args bin and offsets are None, coordinates are not converted to
        the region frame, nor are the projected coords converted back to 
        the original frame. In that case, arg particles is expected to contain
        column self.orig_coord_reg_frame_cols.

        Particles are excluded based on their distance if arg exclusion 
        is not None. The exclusion is performed on each particle set
        (defined using arg set_name_col) separately. 

        Distance exclusion can be done before or after they are 
        projected (depends on arg exclusion_mode). In both cases it
        is done on particle coordinates in the regions frame.

        Arguments:
          - particles (pandas.DataFrame) particles table, like self.particles
          but has to contain data for one tomo only
          - bin: ratio of the original frame and the regions frame bins
          - offsets: offset of the regions frame in respect to the full 
          tomo, in the regions frame coords 
          - class_name_col: column that contains particle set names (classes),
          used to separate particles in individual sets for exclusion
          - region_path: path to the region images, used for projecting
          particles, but not for determining whether particles are inside
          regions (see arg region_shape)
          - region_id: region id (label) in region images 
          - pixel_nm: pixel size in nm
          - project: flag indicating whether to project particle on 
          the region 
          - exclusion: exclusion distance (>0, in nm) or None for no 
          distance based exclusion
          - exclusion_mode: 'before_projection' or 'after_projection', 
          indicate whether exclusion should be aplied on coordinates
          in columns self.orig_coord_reg_frame_cols (before) or in
          self.coord_reg_frame_cols (after particles are projected on
          regions)
          - region_shape: shape of regions image, used to determine
          whether particles are inside this shape
          - inside_coord_cols: column names of coordinates that are
          tested whether they are inside region image shape, if None 
          set to columns self.orig_coord_reg_frame_cols (particles
          in regions frmae before projection to regions)
 
        Returns (pandas.DataFrame) updated particles table
        """

        if (class_name_col is None) and (exclusion is not None):
            raise ValueError(
                "Argument class_name_col has to be specified when "
                + "exclusion is specified.")
        
        coords_orig = particles[self.orig_coord_cols].values
         
        # convert coords to region frame (bin and offset)
        if (bin is not None) and (offsets is not None):
            coords_bin = bin * coords_orig
            coords_bin_off = coords_bin - offsets.reshape(1, -1)
            coords_bin_off = np.rint(coords_bin_off).astype(int)
            particles[self.orig_coord_reg_frame_cols] = coords_bin_off
        
        # exclude before projection 
        if ((exclusion is not None) and (exclusion_mode == 'before_projection')
            and (particles.shape[0] > 0)):
            keep_values = self.exclude_one(
                particles=particles, coord_cols=self.orig_coord_reg_frame_cols,
                class_name_col=class_name_col,
                exclusion=exclusion, pixel_nm=pixel_nm)

        # get region coords and project
        if project:
            # to be replaced by self.project_one()
            if particles.shape[0] > 0:
                region = pyto.segmentation.Labels.read(
                    file=region_path, memmap=True)
                region_coords = get_region_coords(
                    region=region, region_id=region_id, shuffle=False)
                dist = cdist(
                    particles[self.orig_coord_reg_frame_cols].to_numpy(),
                    region_coords)
                try:
                    min_inds = dist.argmin(axis=1)
                except ValueError:
                    raise("Projection failed because region does not exist")
                coords_proj_reg_frame = region_coords[min_inds]
            else:
                coords_proj_reg_frame = np.array([]).reshape(0, 3)
            particles[self.coord_reg_frame_cols] = coords_proj_reg_frame
        else:
            try:
                particles[self.coord_reg_frame_cols] = particles[
                    self.orig_coord_reg_frame_cols]
            except KeyError:
                pass
            
        # convert projected coords back to original frame
        if project and (bin is not None) and (offsets is not None):
            if particles.shape[0] > 0:
                coords_proj_init_frame = (
                    (coords_proj_reg_frame + offsets.reshape(1, -1)) / bin)
                coords_proj_init_frame = (coords_proj_init_frame
                                          .round().astype(int))
            else:
                coords_proj_init_frame = np.array([]).reshape(0, 3)
            particles[self.coord_init_frame_cols] = coords_proj_init_frame
        else:
            try:
                particles[self.coord_init_frame_cols] = particles[
                    self.orig_coord_cols]
            except KeyError:
                pass
            
        # exclude after projection 
        if ((exclusion is not None) and (exclusion_mode == 'after_projection')
            and (particles.shape[0] > 0)):
            keep_values = self.exclude_one(
                particles=particles, coord_cols=self.coord_reg_frame_cols,
                class_name_col=class_name_col,
                exclusion=exclusion, pixel_nm=pixel_nm)

        # find particles inside regions
        if inside_coord_cols is None:
            inside_coord_cols = self.orig_coord_reg_frame_cols
        if region_shape is not None:
            inside_region = self.find_inside_one(
                particles=particles, coord_cols=inside_coord_cols,
                shape=region_shape)
            
        # combine exclusion and find particles, make sure self.keep_col is added
        if particles.shape[0] > 0:
            if exclusion is not None:
                if region_shape is not None:
                    particles[self.keep_col] = keep_values & inside_region
                else:
                    particles[self.keep_col] = keep_values
            else:
                if region_shape is not None:
                    particles[self.keep_col] = inside_region
                else:
                    particles[self.keep_col] = True
        else:
            particles[self.keep_col] = []

        return particles

    def convert_frame(
            self, init_coord_cols, final_coord_cols, shift_final_cols,
            init_bin_col, final_bin_col, to_int=True, overwrite=False):
        """Convert coordinates from init to final frame.

        Conversion is done as follows:
          final_coords = init_coords * init_bin / final_bin - shift_final

        The calculated final frame coordinates are saved in
        final_coord_cols of self.particles. If arg overwrite is False 
        and at least one of the final_coord_cols already exists, 
        ValueError is raised. Otherwise, final_coord_cols columns are 
        overwritten.

        Arguments:
          - init_coord_cols: initial coordinate columns
          - final_coord_cols: final coordinate columns 
          - shift_final_cols: shift of the final frame with respect to
          the init frame at final bin
          - init_bin_col, final_bin_col: initial and final frame bin
          factors, respectively
          - to_int: flag indicating whether final coords are rounded 
          and converted to int
          - overwrite: flag indicating whether overwriting existing 
          final_coord_cols is allowed
        """

        # check if writing new columns is fine
        if not overwrite:
            if len(np.intersect1d(
                    self.particles.columns.to_numpy(), shift_final_cols)) > 0:
                raise ValueError(
                    f"At least one of the final columns ({final_coord_cols})"
                    + "is already present in self.particles and "
                    + "(arg) overwrite is False.")
        
        for _, tomo_row in self.tomos.iterrows():

            # get tomo data and bin
            tomo_id = tomo_row[self.tomo_id_col]
            init_bin = tomo_row[init_bin_col]
            final_bin = tomo_row[final_bin_col]
            mag_factor = init_bin / final_bin
            shift_final = tomo_row[shift_final_cols].to_numpy(dtype=float)
            
            # extract particles for the current tomo
            part_one = self.particles[
                self.particles[self.tomo_id_col]==tomo_id].copy()
            init_parts = part_one[init_coord_cols].to_numpy(dtype=float)
            index_one = part_one.index

            # convert and write
            final_parts = init_parts * mag_factor - shift_final[np.newaxis, :]
            if to_int:
                final_parts = np.round(final_parts).astype(int)
            self.particles.loc[index_one, final_coord_cols] = final_parts

        # convert dtype
        if to_int:
            dtype_conv = dict((col, np.dtype(int)) for col in final_coord_cols)
            self.particles = self.particles.astype(dtype=dtype_conv)
    
    def project_one(
            self, particles, region_path, region_id, coord_cols=None,
            project_mode='closest', angle_cols=None, line_reverse=False,
            line_distance=None, line_grid_mode='unit'):
        """Project multiple particles on a region for one tomo.

        The projection can be done in two ways:
          - project_mode='closest': Points are projected to their closest
        region points
          - project_mode='line': Points are projected along a specified line
        and the projection points are obtained as the line - region
        intersection point that is the closest to the points.

        See the docs in pyto.spatial.LineProjection, including project()
        and __init__() for more info about the 'line' mode.

        Arguments:
          - particles: (pandas.DataFrame) coordinates of points from which
          the line projection is made
          - coord_cols: name of columns that contain point coordinates
          - region_coords: (ndarray n_region_points x 3) coordinates of all
          region points (if None, region and region_id has to be specified)
          - region: (pyto.core.Image, ndarray, or file path) region image 
          onto which points are projected (used only if region_coords is None)
          - region_id: id of the region of interest in the region image  
          (used only if region_coords is None)
          - project_mode: projection mode, 'closest' or 'line'
          - angle_cols: colum names that contain relion tilt and psi
          angles, normally ['rlnAngleTilt', 'rlnAnglePsi'], used only 
          in 'line' projection mode
         - line_grid_mode: 'nearest' or 'unit', determines conversion from 
          line points (floats) to the Cartesian grid (ints)
          - line_reverse: flag indicating if the direction of the resulting 
          line should be reversed, used only in 'line' projection mode
          - line_distance: (single number or an array) projection 
          distance(s) [pixel], all of these are used for each particle.
 
        Returns (ndarray n_points x 3, int) Coordinates of the projected 
        points, or [-1, -1, -1] for each point for which projection could 
        not be determined.
        """

        # get particle coordinates
        if coord_cols is None:
            coord_cols = self.orig_coord_reg_frame_cols
        points = particles[coord_cols].to_numpy()

        if project_mode == 'line':
            if angle_cols is None:
                angle_cols = ['rlnAngleTilt', 'rlnAnglePsi']
            line_projection = LineProjection(
                relion=True, reverse=line_reverse, intersect_mode='first',
                grid_mode=line_grid_mode)
            angles = particles[angle_cols].to_numpy()
            distance = line_distance
        else:
            line_projection = None
            angles = None
            distance = None
            
        # project    
        projected = project(
            points=points, region=region_path, region_id=region_id,
            project_mode=project_mode, line_projection=line_projection,
            line_project_angles=angles, line_project_distance=distance,
            shuffle=False)

        return projected
    
    def exclude(
            self, particles, coord_cols, exclusion, class_name_col,
            pixel_nm=None, tomos=None, set_names=None, consider_keep=False):
        """Distance exclusion of particles of multiple tomos and sets.
   
        Excludes particles based on distance (arg exclusion) for each
        tomo and particle set combination separately (see exclude_one()
        and exclude_single() for more details).

        If pixel size (arg pixel_nm) is specified, that value is used for 
        all tomos and particles. If it is None, pixel size is read from 
        column self.pixel_nm. These values can differ between tomo / particle
        set combinations, but need to be the same for all particles of
        the same tomo and particle set.

        Arguments:
          - particles (pandas.DataFrame) particles table, like self.particles,
          expected to contain particles of one tomo and one particle set
          - tomos: (list) ids of the tomo that should processed (expected to
          be in self.tomo_id_col column), None for all tomos
          - set_names: (list) particle set names that should be processed, 
          None for all particle sets
          - class_name_col: name of the particles colum that containe 
          particle set names
          - coord_cols: (list) name of columns (of particles) containing 
          coordinates used for the exclusion
          - exclusion: exclusion distance in nm
          - pixel_nm: pixel size in nm
          - consider_keep: if True, only particles for which self.keep_col
          column values are True are considered for exclusion and may be
          Flagged True in the returned series.

        Returns (pandas.Series, length equal to the number of particles) flags
        indicating whether particles are retained after exclusion. The
        indices of the returned series directly correspond to the indices
        of the arg particles. Furthermore, the order of indices is the 
        same. If there are no particles, pandas.Series([], dtype=bool) 
        is returned. 
        """

        # set tomos and sets if needed
        if tomos is None:
            tomos = particles[self.tomo_id_col].unique()

        keep = pd.Series([], dtype=bool)
        for to in tomos:
            cond = (particles[self.tomo_id_col] == to)
            selected = particles[cond].copy()
            if selected.shape[0] == 0:
                continue
            
            # find and check pixel size
            if pixel_nm is None:
                pixel_nm_all = selected[
                    self.pixel_nm_col].unique()
                if len(pixel_nm_all) == 1:
                    pixel_nm = pixel_nm_all[0]
                else:
                    raise ValueError(
                        f"Particles of tomo {to} have multiple pixel sizes")

            # exclusion for each set separately 
            keep_one = self.exclude_one(
                particles=selected, coord_cols=coord_cols,
                class_name_col=class_name_col, set_names=set_names,
                exclusion=exclusion, pixel_nm=pixel_nm,
                consider_keep=consider_keep)
            keep = pd.concat([keep, keep_one], axis=0, verify_integrity=True)

        # reorder keep index like particles 
        keep = keep.reindex_like(particles)
            
        return keep

    def exclude_one(
            self, particles, coord_cols, exclusion, class_name_col, pixel_nm=1,
            set_names=None, consider_keep=False):
        """Excludes particles of one tomo and multiple sets that are too close.
    
        The exclusion is imposed separately on individual sets (classes)
        (arg class_name_col) of the specified particles together, that 
        is tomo id is not taken into account.

        This method is meant to be used on particles of one tomo and 
        multiple particle sets (class). In this case, arg particles has to
        contain particles of only one tomo.

        To impose exclusion in the case arg particles contains particles 
        of multiple tomos and sets, so that the exclusion is imposed 
        on each tomo and set separately, use self.exclude(). To impose
        exclusion on all particle regardlss of tomo id and set name, use 
        self.exclude_single().


       """
        
        if set_names is None:
            set_names = particles[class_name_col].unique()
        keep = pd.Series([], dtype=bool)
        for set_nam in set_names:

            # select current tomo and set
            cond = (particles[class_name_col] == set_nam)
            selected = particles[cond].copy()
            if selected.shape[0] == 0:
                continue

            # exclude
            keep_one = self.exclude_single(
                particles=selected, coord_cols=coord_cols,
                exclusion=exclusion, pixel_nm=pixel_nm,
                consider_keep=consider_keep)
            keep = pd.concat([keep, keep_one], axis=0)

        return keep
        
    def exclude_single(
            self, particles, coord_cols, exclusion, pixel_nm=1,
            consider_keep=False):
        """Excludes particles based on distance on all particles together.

        The exclusion is imposed on all specified particles together, that 
        is properties like tomo id and particle set (class) are not taken
        into account.

        This method is meant to be used on particles of one tomo and one 
        and one particle set (class). In this case, arg particles has to
        contain particles of only one tomo and one set.

        To impose exclusion in the case arg particles contains particles 
        of multiple tomos and sets, so that the exclusion is imposed 
        on each tomo and/or set separately, use self.exclude() and 
        self.exclude_one().

        Flags (excludes) particles that are closer to 
        each other than the specified exclusion distance (arg exclusion).
        This results in a set of non-excluded particles where none of them
        is closer to another one than the exclusion distance. 

        More precisely, when two particles are closer to each other than 
        the exclusion, the second one is excluded (according to the row 
        order in arg particles). When within a group of particles several 
        pairs are closer than the exclusion, searches for a way to exclude
        as few particles as possible (see point_pattern.exclude() doc for 
        mode='fine' for more details.)

        It is expected that arg particles contains particles of one tomo
        and one particle set, because the exclusion is based solely on 
        coordinates (arg coord_cols) and does not consider tomo id or 
        particle class columns. 

        Particle coordinates are expected to be in pixels, while (arg)
        exclusion is in nm and is converted to pixels using arg pixel_nm.

        If arg consider_keep is False, all specified particles are 
        consided for exclusion. If column self.keep_col exists in
        the specified particles, it is ignored.

        If arg consider_keep is True, only particles where 
        self.keep_col is True are considered for exclusion. The
        corresponding values in the resulting series are assigned
        as described above, while the values of the resulting series
        where self.keep_col is False are set to False.  

        If arg exclusion is None, it returns pandas.series with all True
        elements in case there is no keep column (self.keep_col ) or arg 
        consider_keep is False. In case arg consider_keep is True,
        the particles keep column is returned.

        Arguments:
          - particles (pandas.DataFrame) particles table, like self.particles,
          expected to contain particles of one tomo and one particle set
          - coord_cols: (list) name of columns (of particles) containing 
          coordinates used for the exclusion
          - exclusion: exclusion distance in nm
          - pixel_nm: pixel size in nm
          - consider_keep: if True, only particles for which self.keep_col
          column values are True are considered for exclusion and may be
          Flagged True in the returned series.

        Returns (pandas.Series, length equal to the number of particles) flags
        indicating whether particles are retained after exclusion. The
        indices of the returned series directly correspond to the indices
        of the arg particles. If there are no particles, pandas.Series([], 
        dtype=bool) is returned. 
        """

        # no particles or no exclusion
        if (particles is None) or particles.empty:
            return pd.Series([], dtype=bool)
        if exclusion is None:
            if (self.keep_col not in particles) or not consider_keep:
                # no or ignore keep column
                return pd.Series(True, index=particles.index)
            else:
                return particles[self.keep_col]

        # get coords that should be retained (not excluded)
        exclusion_pix = exclusion / pixel_nm
        if consider_keep and (self.keep_col in particles):
            particles_local = particles[particles[self.keep_col]]
        else:
            particles_local = particles
        retained_coords = exclude(
            points=particles_local[coord_cols].values, exclusion=exclusion_pix,
            mode='fine')
        if (retained_coords is None) or (retained_coords.size == 0):
            retained = pd.DataFrame(columns=coord_cols+[self.keep_col])
        else:
            retained = pd.DataFrame(retained_coords, columns=coord_cols)
            retained[self.keep_col] = True

        # find particles corresponding to retained coords 
        particles_keep = pd.merge(
            particles, retained, on=coord_cols, how='left',
            suffixes=('_old', '')).set_index(particles.index)
        if self.keep_col not in particles:
            pass  # no keep column to start with 
        elif not consider_keep:
            # ignore previous keep column
            particles_keep = particles_keep.drop(columns=self.keep_col+'_old')
        else:
            # previously keep = False stay False, keep = True updated
            particles_keep = particles_keep.fillna(value={self.keep_col: False})
            particles_keep = particles_keep.drop(columns=self.keep_col+'_old')

        # make sure duplicate rows are removed, may happen for two particles
        # with same coordinates
        duplicated = particles_keep.duplicated(subset=coord_cols, keep='first')
        keep_values = particles_keep[self.keep_col] & ~duplicated

        return keep_values

    def find_inside(
            self, coord_cols, shape_cols, tomos=None, particles=None,
            tomo_ids=None):
        """Find particles that are inside images.

        Finds particles whose coordinates (specified by arg coord_cols)
        are inside tomo-dependent image shapes (specified by arg
        shape_cols in tomos).

        As a common usage example, to find particles that are inside
        regions images before they are projected, use:
          coord_cols=self.coord_reg_frame_cols and 
          shape_cols=self.region_shape_cols

        Arguments:
          - tomos: (pd.DataFrame) tomos table, if None self.tomos is used 
          - tomo_ids: (list) tomo ids
          - particles: (pd.DataFrame) particles table, if None 
          self.particles is used 
          - coord_cols: column names of particles that hold coordinates
          - shape_cols: column names of tomos that hold image shapes

        Returns Sequence of bools, indexed like particles.
        """
        
        # set arguments if needed
        if tomos is None:
            tomos = self.tomos
        if particles is None:
            particles = self.particles
        if tomo_ids is None:
            tomo_ids = particles[self.tomo_id_col].unique()
            
        inside = pd.Series([], dtype=bool)
        for to in tomo_ids:
            cond = (particles[self.tomo_id_col] == to)
            selected = particles[cond].copy()
            if selected.shape[0] == 0:
                continue
            shape = tomos.loc[
                tomos[self.tomo_id_col] == to, shape_cols].to_numpy()

            #shape = tomos[shape_cols].to_numpy()
            inside_one = self.find_inside_one(
                particles=selected, coord_cols=coord_cols, shape=shape)
            inside = pd.concat([inside, inside_one], axis=0)
            
        # reorder keep index like particles 
        inside = inside.reindex_like(particles)
            
        return inside
    
    def find_inside_one(self, particles, coord_cols, shape):
        """Find particles that are inside an image shape for one tomo.

        Returns Sequence of bools, indexed like particles.
        """

        parts = particles[coord_cols].to_numpy()
        lower_ok = np.logical_and.reduce(parts >= 0, axis=1)
        upper_ok = np.logical_and.reduce(
            parts < np.array(shape).reshape(1, -1), axis=1)
        ok = lower_ok & upper_ok
        res = pd.Series(ok, index=particles.index)

        return res
    
