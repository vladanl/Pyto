"""
Methods that convert MultiParticleSets to and from other particle set types.

The other particle set types include ParticleSets, Pyseg, star files
and coordinate tables.
 
# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

import abc
import os

import numpy as np
import pandas as pd

import pyto
from pyto.io.pandas_io import PandasIO
from pyto.particles.relion_tools import get_array_data
import pyto.spatial.coloc_functions as col_func
from .point_pattern import get_region_coords, exclude, project
from .particle_sets import ParticleSets 


class MPSConversion(abc.ABC):
    """Abstract class that provides conversion methods.

    Meant to be inherited by MultiParticleSets.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    def make(self, tomo_star, particle_star, mode, tomo_ids=None,
            pixel_size_nm=None, coord_bin=None,
            region_offset=None, region_bin=None, region_id=None,
            do_origin=True, project=True, exclusion=None,
            exclusion_mode='before_projection',
             class_name='', class_number=None, out_path=None,
             keep_star_coords=True):
        """Generates multi particle sets from different sources.

        Currently implemented only for pyseg / relion generated picks, see
        make_pyseg() doc.
        """

        if mode == 'pyseg':
            self.make_pyseg(
                tomo_star=tomo_star, particle_star=particle_star,
                tomo_ids=tomo_ids, pixel_size_nm=pixel_size_nm,
                coord_bin=coord_bin, region_offset=region_offset,
                region_bin=region_bin, region_id=region_id,
                do_origin=do_origin, project=project, exclusion=exclusion,
                exclusion_mode=exclusion_mode,
                class_name=class_name, class_number=class_number,
                out_path=out_path, keep_star_coords=keep_star_coords)

        else:
            raise ValueError(f"Mode {mode} was not understood")

    def make_pyseg(
            self, tomo_star, particle_star, tomo_ids=None,
            pixel_size_nm=None, coord_bin=None,
            region_offset=None, region_bin=None, region_id=None,
            do_origin=True, project=True, exclusion=None,
            exclusion_mode='before_projection',
            class_name='', class_number=None, out_path=None,
            keep_star_coords=True,
            remove_outside_region=False, inside_coord_cols=None):
        """Generates multi particle sets from pyseg-format star files. 

        Reads pyseg / relion generated particle picks, that is star files that
        contain info about particle coordinates and tomos (args tomo_star
        and particle_star). The corrdinates are given in the initial tomo
        frames (usually the one where particles were picked from Morse
        density tracing).

        The star files pointed by args tomo_star and particle_star should 
        conform to the format used in pyseg processing. Specifically:
          - tomo star contains a row for each tomo
          - pyseg format particle star contains one or more rows, where
          each row has a path to a pyseg/relion particle picks star file 
          that contains particle coordinates

        These initial coordinates are converted to another frame, that is 
        tomograms where membrane regions are defined (typically segmentation
        tomos, which are subtomos of the original tomos and can have different
        binning). Furthermore, the picks may be projected on the region (arg
        project) and the min exclusion distance pitween picks may be
        imposed (arg exclusion).

        The resulting tomos and particles tables are saved as attributes of 
        this instance and may be saved (arg out_path).

        Arguments:
          - tomo_star: path to star file that describes tomograms
          - particle_star: pyseg-format particle star file
          - tomo_ids: tomo ids
          - coord_bin: (single number or list corresponding to tomos) binning
          factor of the system in which particle coordinats are given

          - inside_coord_cols: column names of coordinates that are
          tested whether they are inside region image shape, if None 
          set to columns self.orig_coord_reg_frame_cols (particles
          in regions frmae before projection to regions)

        Sets attributes:
          - tomos
          - particles
        """

        # read tomo star file
        tomos = self.read_star(
            path=tomo_star, mode='tomo', tomo_ids=tomo_ids,
            pixel_size_nm=pixel_size_nm, coord_bin=coord_bin)

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
            
        # read particle star file
        particles_star_df = pd.DataFrame(get_array_data(
            starfile=particle_star, tablename='data', types=str))

        # read all particle star files
        particles = pd.DataFrame()
        for part_star in particles_star_df[self.particle_star_label]:
            if not os.path.isabs(part_star):
                part_star_path = os.path.join(
                    os.path.dirname(particle_star), part_star)
                part_star_path = os.path.normpath(part_star_path)
            else:
                part_star_path = part_star
            particles_local = self.read_star(
                path=part_star_path, mode='particle', tomos=tomos,
                tomo_ids=tomo_ids, do_origin=do_origin,
                class_name=class_name, class_number=class_number)
            try:
                particles = pd.concat(
                    [particles, particles_local], ignore_index=True)
            except (TypeError, NameError):
                particles = particles_local
                
        # convert, project, exclude, find if inside regions image
        # Note: other convert args are read from tomos
        particles = self.convert(
            tomos=tomos, particles=particles, project=project,
            exclusion_mode=exclusion_mode,
            remove_outside_region=remove_outside_region,
            inside_coord_cols=inside_coord_cols)

        # save
        self.tomos = tomos
        self.particles = particles
        if out_path is not None:
            self.write(out_path)

    def from_particle_sets(
            self, particle_sets, columns=None, region_id=None,
            pixel_to_particles=False):
        """Converts pyto.spatial.ParticleSets instance to this class.

        Takes particle sets specified as an instance of ParticleSets or 
        pandas.DataFrame (such as particle_sets_instance.data_df) and extract
        information to make tomos and particles table of this instance.

        If (arg) particle_sets has standard labels, the default conversion from 
        labels of particle_sets and this instance is used:
          default_conversion = {
            'tomo_id': self.tomo_id_col, 'set': self.class_name_col, 
            'region_path': self.region_col, 'pixel_nm': self.pixel_nm_col,
            'x': self.coord_reg_frame_cols[0],
            'y': self.coord_reg_frame_cols[1],
            'z': self.coord_reg_frame_cols[2]}

        In case (arg) particle_sets have non-standard labels, arg column has
        to be specified to define label conversion from particle_sets to
        this instance. In this case, the default conversion is updated with 
        (arg) columns.

        Sets attributes:
          - tomos: tomogram table
          - particles: particles table

        Arguments:
          - particle sets: (ParticleSets or pandas.DataFrame) input particle 
          sets
          - columns: (dict) Mapping between column names of the input particle
          sets (keys) and output (this instance) tables (values)
          - region_id: if not None, the value is added as column 
          self.region_id_col to all rows of self.tomos 
          - pixel_to_particles: flag indicating if pixel size in nm is added
          to particles table (it is always added to tomos table)
        """

        # setup column label renaming
        ps_0 = ParticleSets()
        columns_default = {
            ps_0.tomo_col: self.tomo_id_col,
            ps_0.set_name_col: self.class_name_col, 
            ps_0.region_path_col: self.region_col,
            ps_0.pixel_col: self.pixel_nm_col,
            ps_0.coord_cols[0]: self.coord_reg_frame_cols[0],
            ps_0.coord_cols[1]: self.coord_reg_frame_cols[1],
            ps_0.coord_cols[2]: self.coord_reg_frame_cols[2]}
        if columns is not None:
            columns_default.update(columns)
        conversion = columns_default

        # rename ParticleSets columns
        if isinstance(particle_sets, ParticleSets):
            ps_df = particle_sets.data_df
        else:
            ps_df = particle_sets
        conversion_keys = list(conversion.keys())
        for col in conversion_keys:
            if col not in ps_df.columns:
                conversion.pop(col)
        ps_df_2 = ps_df.rename(columns=conversion)

        # make tomos table
        self.tomos = (ps_df_2[
            [self.tomo_id_col, self.region_col, self.pixel_nm_col]]
                     .drop_duplicates()
                     .reset_index(drop=True))
        if len(self.tomos[self.tomo_id_col].unique()) != self.tomos.shape[0]:
            raise ValueError(
                "ParticleSets have to have unique values of region_path and "
                "pixel size for each tomogram.")

        # add region id
        if region_id is not None:
            self.tomos[self.region_id_col] = region_id
        
        # make particles
        if pixel_to_particles:
            particle_drop = [self.region_col]
        else:
            particle_drop = [self.region_col, self.pixel_nm_col]
        self.particles = ps_df_2[ps_df_2.columns.drop(particle_drop)].copy()
        self.particles[self.class_number_col] = -1
        self.particles[self.keep_col] = True
        cols = (self.particles.columns
                .drop(self.class_number_col)
                .insert(item=self.class_number_col, loc=2))
        self.particles = self.particles[cols]

    def to_particle_sets(
            self, columns=None, set_name_col=None, coord_cols=None,
            ignore_keep=False, index=True):
        """Converts data of this instance to ParticleSets.

        Makes a new instance of ParticleSets.

        Arg columns defines how column names of this instance (keys)
        are converted to column names of ParticleSets.data_df table (values).
        If None, the default values are used, which is fine if the 
        relevant attributes of this instance (defined in __init__()) were
        not changed from their default values.

        By default, columns from this instance are mapped to columns
        of the resulting ParticleSets instance (psets) as follows:
            self.tomo_id_col: psets.tomo_col
            self.region_col: psets.region_path_col
            self.pixel_nm_col: psets.pixel_col
        In addition the following mapping as applied:
            (arg) set_name_col: psets.set_name_col
        These are typically sufficient. 

        However, if arg columns is specified, the default dictionary 
        is updated with the specified dictionary.

        Setting arg index to True, guarantees that the resulting 
        particle_sets.data_df will have the same index as self.particles.
        However, the rows might not be in the same order. To achive this use:
        particle_sets.data_df.sort_index().

        Arguments:
          - columns: (dict) Defines how column names of this instance (keys)
          are converted to column names of ParticleSets.data_df table (values) 
          - set_name_col: specifies which column of self.particles is
          converted to set name column of the resulting ParticleSets, if None
          self.class_name_col is used
          - coord_cols: (list) Specifies which (coordinate) columns of this
          instance hold the coorfinates that are transfered to the resulting
          ParticleSets instance (default self.coord_reg_frame_cols)
          - ignore_keep: (bool) If False, only the coordinates that have
          True in the keep column (self.keep_col) are passed to the resulting
          ParticleSets instance
          - index: flag indicating whether index of self.particles is passed
          to the resulting ParticleSets
          
        Returns: instance of ParticleSets
        """

        # setup column label renaming
        psets = ParticleSets()
        columns_default = {
            self.tomo_id_col: psets.tomo_col,
            self.region_col: psets.region_path_col,
            self.pixel_nm_col: psets.pixel_col}
        if set_name_col is None:
            set_name_col_local = self.class_name_col
            columns_default[self.class_name_col] = psets.set_name_col
        else:
            set_name_col_local = set_name_col
            columns_default[set_name_col] = psets.set_name_col
        if coord_cols is None:
            coord_cols = self.coord_reg_frame_cols
        coord_cols_default = dict(zip(coord_cols, psets.coord_cols))
        coord_col_keys = list(coord_cols_default.keys())
        for ccol in coord_cols_default:
            if ccol not in self.particles.columns:
                coord_cols_default.pop(ccol)
        columns_default.update(coord_cols_default)
        conversion = columns_default.copy()
        if columns is not None:
            conversion.update(columns)

        # extract data from this instance that is needed for ParticleSets
        if (not ignore_keep) and (self.keep_col in self.particles.columns):
            particles_loc = self.particles[self.particles[self.keep_col]].copy()
        else:
            particles_loc = self.particles
        part_cols = [self.tomo_id_col, set_name_col_local] + coord_cols
        part_cols = [col for col in part_cols if col in self.particles.columns]
        #if index:
        #    particles_loc[psets.index] = particles_loc.index
            #part_cols = [psets.index_col] + part_cols
        tomo_cols = [self.tomo_id_col, self.region_col, self.pixel_nm_col]
        psets_df = pd.merge(
            particles_loc[part_cols], self.tomos[tomo_cols],
            on=self.tomo_id_col, how='left', sort=False)
        if index:
            psets_df.index = particles_loc.index

        # convert labels
        psets_df = psets_df.rename(columns=conversion)
        psets.data_df = psets_df

        return psets
        
    def from_coords(
            self, particle_df, coord_cols, tomo_id_col, tomo_star=None,
            tomo_ids=None, pixel_size_nm=None, coord_bin=None,
            region_offset=None, region_bin=None, region_id=None,
            col_conversion={}, project=True,
            exclusion=None, exclusion_mode='before_projection',
            remove_outside_region=False, inside_coord_cols=None,
            particle_name='', out_path=None):
        
        """Converts coordinates from an arbitrary DataFrame to this class.

        Reads tomo data from the specified pyseg/relion type star file
        (arg tomo_star) and adds columns from args pixel_size_nm, 
        coord_bin, region_offset, region_bin, region_id and exclusion,
        if these exist. This data is saved as self.tables.

        Reads particle data from the specified (path to a) dataframe  
        (arg particle_df). This (input) dataframe has to contain 
        
        Similar to 

        Arguments:

          - coord_cols: column names of arg particle_df that contain
          particle coordinates; these are saved in column 
          self.orig_coord_cols of self.particles          

          - inside_coord_cols: column names of coordinates that are
          tested whether they are inside region image shape, if None 
          set to columns self.orig_coord_reg_frame_cols (particles
          in regions frame before projection to regions)

        """

        # read particle coords df from a file, if needed
        if isinstance(particle_df, str):
            particle_df = PandasIO.read(base=particle_df)

        # convert column names and add columns
        coord_conversion = dict(zip(coord_cols, self.orig_coord_cols))
        coord_conversion[tomo_id_col] = self.tomo_id_col
        if col_conversion is None:
            col_conversion = {}
        col_conversion.update(coord_conversion)
        particles = particle_df.rename(columns=col_conversion)
        particles[self.class_name_col] = particle_name    
        if pixel_size_nm is not None:
            particles[self.pixel_nm_col] = pixel_size_nm

        # select tomos, if needed
        if tomo_ids is not None:
            particles = particles[
                particles[self.tomo_id_col].isin(tomo_ids)]

        # read tomo star file
        if tomo_star is not None:
            tomos = self.read_star(
                path=tomo_star, mode='tomo',  tomo_ids=tomo_ids,
                pixel_size_nm=pixel_size_nm, coord_bin=coord_bin)
        else:
            tomos = self.make_bare_tomos(
                particles=particles, pixel_size_nm=pixel_size_nm)
 
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
        if self.region_col in tomos:
            abs_dir = os.path.abspath(os.path.dirname(tomo_star))
            self.get_region_shapes(tomos=tomos, curr_dir=abs_dir, update=True)
        elif remove_outside_region:
            raise ValueError(
                f"Arg remove_outside_region is True, but tomos table does not "
                + f"have region path (column {self.region_col})")  
             
        # project and exclude
        # Note: other convert args are read from tomos
        particles = self.convert(
            tomos=tomos, particles=particles, project=project,
            exclusion_mode=exclusion_mode,
            remove_outside_region=remove_outside_region,
            inside_coord_cols=inside_coord_cols)
                    
        # save
        self.tomos = tomos
        self.particles = particles
        if out_path is not None:
            self.write(out_path)
        
    def from_coords_df(
            self, particle_df, tomo_star=None, tomo_ids=None,
            col_conversion=None,
            pixel_size_nm=None, 
            particle_name='', particle_classes={},
            particle_class_dfs={}):
        
        """Converts arbitrary DataFrame coordinates to particles table (old).

        Differs from from_coords() in that it does not convert particle
        coords to the regions frame and does not project them to regions.

        Likely depreciated in favour of from_coords().
        """

        # read tomo star file
        if tomo_star is not None:
            self.tomos = self.read_star(
                path=tomo_star, mode='tomo',  tomo_ids=tomo_ids,
                pixel_size_nm=pixel_size_nm)
        
        # read particle coords df from a file, if needed
        if isinstance(particle_df, str):
            particle_df = PandasIO.read(base=particle_df)
        if tomo_ids is not None:
            particle_df = particle_df[
                particle_df[self.tomo_id_col].isin(tomo_ids)]

        # convert and add columns
        if col_conversion is not None:
            particle_df = particle_df.rename(columns=col_conversion)
        particle_df[self.class_name_col] = particle_name    
        particle_df[self.class_number_col] = -1
        particle_df[self.pixel_nm_col] = pixel_size_nm

        # add particle class number
        for p_class, p_num in particle_classes.items():

            # get coordinates
            p_class_df = particle_class_dfs[p_class]
            if isinstance(p_class_df, str):
                p_class_df = PandasIO.read(base=p_class_df)

            # convert
            if col_conversion is not None:
                p_class_df = p_class_df.rename(columns=col_conversion)
                p_class_df[self.class_number_col] = p_num

            # add class number to the main particle df
            local_cols = ([
                self.tomo_id_col, self.particle_id_col, self.class_number_col]
                + self.orig_coord_cols)
            particle_df = pd.merge(
                particle_df, p_class_df[local_cols],
                on=[
                    self.tomo_id_col,
                    self.particle_id_col] + self.orig_coord_cols,
                suffixes=('', '_new'), how='left', sort=False)
            cn_new_col = self.class_number_col + '_new'
            class_new = particle_df[self.class_number_col].where(
                particle_df[cn_new_col].isna(), particle_df[cn_new_col])
            particle_df[self.class_number_col] = class_new.to_numpy()
            particle_df = particle_df.drop(cn_new_col, axis=1)
            
        return particle_df

    def from_patterns(
            self, patterns, coord_cols, tomo_id, pixel_size_nm=1, update=False):
        """Makes a basic particles table from point patterns for one tomo.

        Makes table with following columns:
          - self.tomo_id_col: tomo id
          - self.pixel_nm_col: pixel size 
          - self.class_name_col: pattern name
          - self.subclass_col: pattern_name
          - self.keep_col: set to True

        Arguments:
          - patterns: (dict pattern_name : ndarray N_points x ndim) point 
          pattern (particles) of one tomo, all patterns have to have the 
          same dtype
          - coord_cols: (list) coordinate column names 
          - tomo_id: tomo id
          - pixel_size_nm: pixel size in nm (default 1)
          - update: if True, adds the resulting table to self.particles
          (default False)

        Returns: 
          - if update is False: (pandas.DataFrame) particles table
          - if update is True modifies self.particles
        """

        # particles from patterns
        part_dfs = []
        for name, pat in patterns.items():
            part_local = pd.DataFrame(
                {self.tomo_id_col: tomo_id, self.pixel_nm_col: pixel_size_nm,
                 self.class_name_col: name, self.subclass_col: name, 
                 **dict(zip(coord_cols, pat.transpose())),
                 self.keep_col: True})
            part_dfs.append(part_local)
        result = pd.concat(part_dfs, ignore_index=True)

        # update or return
        if update:
            try:
                self.particles = pd.concat(
                    [self.particles, result], ignore_index=True)
            except AttributeError:
                self.particles = result
        else:
            return result
    
    def from_star_picks(
            self, particle_star, tomo_star=None,
            tomo_ids=None, tomo_id_mode='munc13',
            pixel_size_nm=None, coord_bin=None, region_path=None,
            region_offset=None, region_bin=None, region_id=None,
            do_origin=True, convert=True, project=True, exclusion=None,
            exclusion_mode='before_projection',
            class_name='', class_number=None, 
            out_path=None, keep_star_coords=True,
            remove_outside_region=False, inside_coord_cols=None):
        """Makes an instance from pyseg/relion particle picks star file.

        Makes an instance of this class in the same way as make_pyseg().
        The important difference is that arg particle_star has to
        be a path to pyseg/relion particle picks star file (and not to 
        a file that contains paths to these files).

        Also, arg tomo_star does not need to be specified, in which case, 
        a bare-bones tomos table is generated (see make_bare_tomos()). 
        As a consequence, tomos table will not contain paths to region
        images, which precludes projection to regions.

        Arguments:
          - particle_star: path to particles star file, or (experimental)
          particles table (pandas.Dataframe) 
          - tomo_star: path to tomos star file, or (experimental)
          tomos table (pandas.Dataframe) in the form obtained by 
          read_star(path=tomo_star_path, ...)
          - tomo_id_mode: mode for determining tomo id, needed if both
          tomo_star and tomo_ids are None to extract tomo ids from
          particles_star; passed directly to 
          coloc_functions.get_tomo_id(mode)
          - pixel_size_nm: (single number, dict where keys are tomo ids, or 
          pandas.DataFrame having tomo_id column) pixel size [nm] of the 
          system in which particle coordinats are given
          - coord_bin: (single number, dict where keys are tomo ids, or 
          pandas.DataFrame having tomo_id column) binning
          factor of the system in which particle coordinats are given,
          needed in mode 'tomo'
          - region_path: region path, used exceptionally, only when 
          tomo_star is None and both convert and project are True
          - region_bin: bin factor of the region images 
          - region offset: offset of region images relative to the initial
          tomo frame, specified as expalined in set_region_offset() doc.
          - region_id: region id in region images
          - convert: flag indicating whether coordinates are converted
          to regions frame
          - project: flag indicating whether coordinates are projected
          on regions after coversion
          - exclusion: exclusion distance (>0, in nm) or None for no 
          distance based exclusion
          - exclusion_mode: distance exclusion is applied on particles
          before the projection (after conversion) if 'before_projection',
          or after the projection if 'after_projection'
          - class_name: name of the classification
          - class_number: class number
          - out_path: path where this instance is written
          - keep_star_coords: If True, star file coordinates and offsets
          (columns rlnCoordinateX/Y/Z and rlnOriginX/Y/Z or 
          rlnOriginX/Y/ZAngst are saved in the resulting particles table
          - remove_outside_region: flag indicating whether coordinates
           that fall outside region images after conversion are excluded
           - inside_coord_cols: column names of coordinates that are
           tested whether they are inside region image shape, if None 
           set to columns self.orig_coord_reg_frame_cols (particles
           in regions frmae before projection to regions)

        """

        # get tomos table
        tomo_star_path_exists = False
        if tomo_star is not None:

            if isinstance(tomo_star, pd.DataFrame):
                tomos = tomo_star  # tomos star read already
            else:
                # read tomo star file
                tomos = self.read_star(
                    path=tomo_star, mode='tomo', tomo_ids=tomo_ids,
                    pixel_size_nm=pixel_size_nm, coord_bin=coord_bin)
                tomo_star_path_exists = True

        else:

            # get tomo ids from particle star (read for real later)
            if tomo_ids is None:
                particles_tmp = pd.DataFrame(get_array_data(
                    starfile=particle_star, tablename='data', types=str))
                particles_tmp[self.tomo_id_col] = (
                    particles_tmp[self.micrograph_label]
                    .map(
                        lambda x: col_func.get_tomo_id(
                            path=x, mode=tomo_id_mode)))
#                tomo_ids = np.sort(
#                    particles_tmp[self.tomo_id_col].unique())
                tomo_ids = particles_tmp[self.tomo_id_col].unique()

            # make tomos table
            tomos = self.make_bare_tomos(
                tomo_ids=tomo_ids, pixel_size_nm=pixel_size_nm)
            if coord_bin is not None:
                tomos = self.set_column(
                    column=self.coord_bin_cols, tomos=tomos, value=coord_bin,
                    update=True)
            if region_path is not None:
                tomos = self.set_column(
                    column=self.region_col, tomos=tomos, value=region_path,
                    update=True)
            else:
                if convert and project:
                    raise ValueError(
                        "Args convert and project are both True, but "
                        + "it is not possible to project because arg "
                        + "region_path is not specified.")

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
        if self.region_col in tomos:
            if tomo_star_path_exists:
                abs_dir = os.path.abspath(os.path.dirname(tomo_star))
            else:
                abs_dir = ''
            self.get_region_shapes(tomos=tomos, curr_dir=abs_dir, update=True)
        elif remove_outside_region:
            raise ValueError(
                f"Arg remove_outside_region is True, but tomos table does not "
                + f"have region path (column {self.region_col})")  

        # read particles
        particles = self.read_star(
            path=particle_star, mode='particle', tomos=tomos,
            tomo_ids=tomo_ids, do_origin=do_origin,
            class_name=class_name, class_number=class_number)
        
        # convert, project, exclude, find if inside regions image
        # Note: other convert args are read from tomos
        if convert:
            particles = self.convert(
                tomos=tomos, particles=particles, project=project,
                exclusion_mode=exclusion_mode,
                remove_outside_region=remove_outside_region,
                inside_coord_cols=inside_coord_cols)

        # save
        self.tomos = tomos
        self.particles = particles
        if out_path is not None:
            self.write(out_path)
       
    def make_bare_tomos(
            self, particles=None, tomo_ids=None, pixel_size_nm=None):
        """Make bare-bones tomos table.

        Makes tomos table that contain the minimum info, that is tomo
        ids and possibly pixel size.

        In order to determine tomo ids, either arg particles or 
        tomo_ids need to be specified.

        Arguments:
          - particles: (pandas.DataFrame) particles table
          - tomo_ids: (list or similar) tomo ids
          - pixel_size_nm: pixel size in nm, see set_colum() doc for 
          possible formats 

        Returns (pandas.DataFrame) tomos table, one row for each tomo
        (like self.tomos).
        """

        # make tomos with tomo ids
        if (particles is not None) and (tomo_ids is not None):
            raise ValueError(
                "Either particles or tomo_ids arg should be specified.")
        elif particles is not None:
            tomo_ids = particles[self.tomo_id_col].unique()
        elif tomo_ids is None:
            raise ValueError(
                 "Either particles or tomo_ids arg should be specified.")
        tomos = pd.DataFrame({self.tomo_id_col: tomo_ids})

        # pixel size
        if pixel_size_nm is not None:
            tomos = self.set_column(
                tomos=tomos, column=self.pixel_nm_col, value=pixel_size_nm,
                update=True)

        return tomos
        
   
