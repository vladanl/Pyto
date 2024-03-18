"""
Methods that convert or relate different MultiParticleSets to each other. 

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

import abc
import os

import numpy as np
import pandas as pd

import pyto
from pyto.util.pandas_plus import merge_left_keep_index
from pyto.io.pandas_io import PandasIO


class MPSInterconversion(abc.ABC):
    """Abstract class that provides conversion methods.

    Meant to be inherited by MultiParticleSets.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    def from_mps(
            self, set_name, resolve_fn, psets_module=None,
            tomo_ids=None, discard_tomo_ids=None,
            subclass_col='subclass_name', ignore_keep=False):
        """Combines multiple particle sets based on particle sets module.

        Reads one or more particles sets (names given by arg set_name) 
        that were previously saved as individual instances of this class 
        and combines (concatenates) them to this instance.

        Intended use is to make an instance of this class that can be used to
        determine colocalization (passed to ColocLite.colocalize()).

        Uses the specified particle sets module (arg psets_module) to 
        resolve paths to saved instances of this class that contain all 
        particles specified by arg set_name (hese may contain additional 
        particles). Then reads the pickles, select only particle and tomos 
        that correspond to the specified particle (sub)sets (arg set_name)
        and makes a temporary insstance of this class for each particle set.

        When more than one set is specified (arg set_name is a list of 
        length >1), tomos and particles tables (attributes) of each 
        individual particle set are combined (concatenated) to make a new 
        instance of this class. When concatenated, only the common 
        columns are kept.

        Furthermore, in the resulting tomos table, all values of column 
        self.coord_bin_col are set to NaN, because the input particles sets 
        (arg set_name) may have coordinates determined at different bins. 
        During the concatenation, duplicate rows of tomos table are removed.

        In the resulting particles table, columns self.tomo_id_col, 
        self.class_name_col (or subclass_col) and self.particle_id_col 
        taken together uniquely specify each particle (row in 
        self.particles table). 

        The particles indices of the sets that are concatenated are saved
        in column self.pick_index_col. Because self.pick_index_col
        values come from different
        tables, they do not need to be unique when put together. However, 
        the combination of self.pick_index_col and self.class_name_col 
        (or subclass_col) columns is unique.

        In the final self.particles table, index is reset (without moving 
        rows) so that particles are also uniquely specified by index.

        Consequently, input particles (rows of the input particle sets) 
        can be matched to the resulting particles (rows of the resuling
        self.particles) as follows:
          - index of input particle set of name 'SetA'
        is the same as:
          - pick_index of self.particles restricted to class name 'SetA'

        Arguments:
          - set_name: (str or list of strs) one or more particle set names
          - psets_module: particle sets module, specified as a module
          or a path to the module
          - resolve_fn: function that resolves particle set names, signature:
            instance, class_name, class_number = resolve_fn(
                set_name, particle_sets_module=None)
          - tomo_ids: ids of tomos to use, or None for all tomos (default)
          - discard_tomo_ids: ids of the tomos that should not be used, if
          not None arg tomo_ids has to be None (default None)
          - subclass_column: column name in particles table that holds 
          subclass names (default 'subclass_name')
          - ignore_keep: If False (default), only particles where 'keep' 
          column is True are selected, True to ignore 'keep' column

        Sets attributes:
          - tomos: tomo table
          - particles: particle table
          - subclass_name_col: set from arg subclass_col
        """

        # check args
        if (tomo_ids is not None) and (discard_tomo_ids is not None):
            raise ValueError(
                "Only one of the arguments tomo_ids and discard_tomo_ids "
                + " can be specified.")
        
        # import particle sets module if needed
        if ((psets_module is not None) and isinstance(psets_module, str)):
            psets_module = ParticleSets.import_module(path=psets_module)

        # loop over sets
        if isinstance(set_name, str):
            set_name = [set_name]
        for set_na in set_name:

            # get particles for this set
            #mps_local, cl_nam, cl_num = resolve_fn(
            #    set_name=set_na, psets_module=psets_module)
            if psets_module is None: 
                resolve_result = resolve_fn(set_name=set_na)
            else:
                resolve_result = resolve_fn(
                    set_name=set_na, psets_module=psets_module)
            if isinstance(resolve_result, self.__class__):
                mps_local = resolve_result
                if discard_tomo_ids is not None:
                    tomo_ids = np.setdiff1d(
                        mps_local.tomos[mps_local.tomo_id_col].to_numpy(),
                        discard_tomo_ids)
                mps_local.select(tomo_ids=tomo_ids, update=True)
            else:
                mps_local, cl_nam, cl_num = resolve_result             
                if discard_tomo_ids is not None:
                    tomo_ids = np.setdiff1d(
                        mps_local.tomos[mps_local.tomo_id_col].to_numpy(),
                        discard_tomo_ids)
                mps_local.select(
                    class_names=[cl_nam], class_numbers=cl_num,
                    tomo_ids=tomo_ids, update=True)
            mps_local.subclass_name_col = subclass_col
            mps_local.particles[mps_local.subclass_name_col] = set_na
            if not ignore_keep:
                mps_local.particles = mps_local.particles[
                    mps_local.particles[mps_local.keep_col]]

            # merge with the existing tomos and particles
            mps_local.tomos[mps_local.coord_bin_col] = np.nan
            try:
                self.tomos = (
                    pd.concat([self.tomos, mps_local.tomos], join='inner')
                    .drop_duplicates())
                if (self.tomos.shape[0]
                    != self.tomos[self.tomo_id_col].unique().shape[0]):
                    raise ValueError(
                        "Tomo ids in self.tomos table are not unique.")
                self.particles = pd.concat(
                    [self.particles, mps_local.particles], join='inner')
            except AttributeError:
                self.tomos = mps_local.tomos
                self.particles = mps_local.particles

        # reset index because particles from different sets may
        # have same indices, and that conflicts with json
        #self.particles = self.particles.reset_index(
        #    names=[self.pick_index_col])  # version 1.5+
        self.particles = self.particles.reset_index().rename(
            columns={'index': self.pick_index_col}) 

    def label_coloc(
            self, coloc, label_col, distance, label_mode='str',
            particles=None, update=False):
        """Labels colocalized particles in input colocalization table.

        Adds a label for each particle of the current table (a row in 
        particles or self.particles) that shows whether the particle 
        is found in a colocalization (arg coloc). 

        The current table (self.particles or particles) has to be 
        an input colocalization table. Arg coloc specifies
        a colocalization result table 

        Depending on the arg label_mode, the label is 'True' or True if it 
        is found and 'False' or False otherwise. Using the string version
        is in some cases more suitable for further processing

        It is essential that indices of the particles and colocalization 
        tables correspond to each other and that the colocalization table
        contains a subset (or all) of particles from table particles. 
        
        """

        # figure out args
        if isinstance(coloc, str):
            coloc_tab = pyto.io.PandasIO.read(coloc)
        elif isinstance(coloc, pd.DataFrame):
            coloc_tab = coloc
        else:
            raise ValueError(
                "Argument coloc has to be a path (str) to, or a DataFrame.")
        if particles is None:
            particles = self.particles.copy()
        if label_mode == 'str':
            true_l = 'True'
            false_l = 'False'
        elif label_mode == 'bool':
            true_l = True
            false_l = False
        else:
            raise ValueError(
                f"Argument label_mode ({label_mode}) has to be "
                + "'str' or 'bool'.")

        # select distance
        coloc_tab = coloc_tab[coloc_tab['distance'] == distance]
        
        # label
        particles[label_col] = false_l
        particles.loc[coloc_tab.index, label_col] = true_l

        if update:
            self.particles = particles
        else:
            return particles

    def add_classification_coloc(
            self, other, particles=None,
            check=False, check_cols=None, update=False, verbose=False):
        """Adds classification columns to a colocalization table.

        This instance is expected to contain particles from colocalization
        results. 

        Like (calls) extend_coloc(), but default arguments are set so that 
        they are appropriate for adding colocalization columns to
        colocalization results particles. see extend_coloc() docs for 
        more info.

        """

        # figure out other, needed to get specific values here
        if isinstance(other, str):
            other_mps = self.__class__.read(other, verbose=verbose)
        elif isinstance(other, self.__class__):
            other_mps = other
        else:
            raise ValueError(
                "Argument other has to be a path (str) to, or an instance "
                + f"of this class (self.__class__.")
        
        other_cols = other_mps.classification_cols
        if check and (check_cols is None):
            check_cols = ([self.tomo_id_col, self.particle_id_col]
                          + self.coord_reg_frame_cols)

        # merge
        result = self.extend_coloc(
            other=other_mps, other_cols=other_cols,
            particles=particles, check=check, check_cols=check_cols,
            update=False, verbose=verbose)

        if update:
            self.classification_cols = other_mps.classification_cols
        else:
            return result

    def extend_coloc(
            self, other, other_cols, particles=None,
            check=False, check_cols=None, update=False, verbose=False):
        """Adds columns from a main particles to a colocalization table.

        This instance is expected to contain particles from colocalization
        results (as self.particles). The particles whose properties are 
        added (arg other.particles) are usually those that are given as 
        initial particles to colocalization (arg particles 
        of ColocLite.colocalize()). In this case, the constraints given 
        below are automatically satisfied. 

        Importantly, other particle sets can be used, but the following
        constraints need  to be satisfied: 
          - Each particle of this instance has to have exactly one 
          corresponding particle in other (but other can contain particles 
          that do not correspond to any particle of this instance). 
          - The correspondence is established so that columns 'pick_index'
          and self.class_name_col of this instance are the same as
          index and other.class_name_col of other particles

        Optionally, to ensure the correspondence is correct, other columns
        of self.particles and other.particles (arg check_cols) are checked 
        to determine whether they have the same values. If they are not
        the same, a warning
        is printed and the checked columns from other.particles (with 
        suffix '_err') are added to the resulting particles table.

        If arg check_cols is None (and check is True) columns checked are
        self.tomo_id_col, self.particle_id_col and 
        self.coord_reg_frame_cols.

        Arguments:
          - other: (MultiParticleSets) particles whose properties are added
          to this instance, or the path to pickled MultiParticleSets
          instance
          - other_cols: (list) columns of other that are added
          - coloc_on: column of self.particles that corresponds to index 
          of other.particles 
          - particles: (DataFrame) if specified, used instaed of 
          self.particles  (default None)
          - check: flag indicationg whether other columns are checked 
          (default False)
          - check_columns: (list) columns checked
          - update: self.particles are updated if True, otherwise the
          modified (extended) particles (DataFrame) are returned
          - verbose: flag related to reading a pickled other particles 

          Returns: extended (modified) particles, if update is False
         """

        # figure out other, needed to get specific values here
        if isinstance(other, str):
            other_mps = self.__class__.read(other, verbose=verbose)
        elif isinstance(other, self.__class__):
            other_mps = other
        else:
            raise ValueError(
                "Argument other has to be a path (str) to, or an instance "
                + f"of this class (self.__class__.")

        # give other particles index a name and setup coloc merge 
        other_index_name = other_mps.particles.index.name
        other_mps.particles.index.name = 'index_tmp'
        other_on = ['index_tmp', other_mps.class_name_col]
        on = ['pick_index', self.class_name_col]

        if check and (check_cols is None):
            check_cols = ([self.tomo_id_col, self.particle_id_col]
                          + self.coord_reg_frame_cols)

        # merge and reset other index name
        parts = self.extend_particles(
            self, other=other_mps, other_cols=other_cols, on=on,
            other_on=other_on, particles=particles,
            check=check, check_cols=check_cols, update=False, verbose=verbose)     
        other_mps.particles.index.name = other_index_name

        if update:
            self.particles = parts
        else:
            return parts

    def extend_particles(
            self, other, other_cols, on, other_on=None, particles=None,
            check=False, check_cols=None, update=False, verbose=False):
        """Adds columns from other to this particles table.

        """
        
        # figure out args
        if isinstance(other, str):
            other_mps = self.__class__.read(other, verbose=verbose)
        elif isinstance(other, self.__class__):
            other_mps = other
        else:
            raise ValueError(
                "Argument other has to be a path (str) to, or an instance "
                + f"of this class (self.__class__.")
        if particles is None:
            particles = self.particles.copy()
        if other_on is None:
            other_on = on
            
        # setup check
        if check:
            other_cols_ext = other_on + other_cols + check_cols
        else:
            other_cols_ext = other_on + other_cols

        # merge and revert other index
        parts = merge_left_keep_index(
            particles, other.particles[other_cols_ext],
            #left_on=coloc_on, right_index=True, suffixes=('', '_y'))
            left_on=on, right_on=other_on, suffixes=('', '_y'))

        # check and clean:
        if check:
            check_cols_y = [col + '_y' for col in check_cols]
            cond = np.logical_and.reduce(
                parts[check_cols_y].notnull().to_numpy(), axis=1)
            check_passed = np.logical_and.reduce(
                (parts.loc[cond, check_cols].to_numpy()
                 == parts.loc[cond, check_cols_y].to_numpy()),
                axis=1)
            if not check_passed.all():
                print(
                    "WARNING: Check failed, coordinates in columns "
                    + f"{check_cols} in the coloclization and other tables "
                    + "are different. Columns from other.particles "
                    + "table are saved with suffix 'err'.")
                rename_check = dict(
                    (col, f'{col}_err') for col in check_cols_y)
                parts.rename(columns=rename_check, inplace=True)
            else:
                parts.drop(columns=check_cols_y, inplace=True)

        if update:
            self.particles = parts
        else:
            return parts


