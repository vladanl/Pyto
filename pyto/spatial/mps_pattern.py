"""
Methods that provide point pattern functionality to MultiParticleSets

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

import abc

import numpy as np
from numpy.random import default_rng
import pandas as pd

from . import point_pattern


class MPSPattern(abc.ABC):
    """Abstract class that provides pattern related methods for MPS.

    Meant to be inherited by MultiParticleSets.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    def translate_pattern(
            self, original, new, distance, tomo_id, coord_cols=None, 
            tomo_id_col=None, update=True):
        """Translates a point pattern and adds it to the patterns table.

        Both class and subclass columns (self.class_name_col and
        self.subclass_col) are set to the name of the translated pattern
        (arg new).

        
        """

        # set defaults
        if coord_cols is None:
            coord_cols = self.coord_reg_frame_cols
        if tomo_id_col is None:
            tomo_id_col = self.tomo_id_col
        #if class_name_col is None:
        #    class_name_col = self.class_name_col

        # get old pattern
        pattern = (
            self.particles
            .query(
                f"{tomo_id_col} == @tomo_id and {self.class_name_col} "
                + f"== @original")
            .copy())

        # translate
        pattern[coord_cols] = pattern[coord_cols] + np.asarray(distance)
        pattern.loc[:, self.class_name_col] = new
        pattern.loc[:, self.subclass_col] = new

        # save or return
        if update:
            self.particles = pd.concat(
                [self.particles, pattern], ignore_index=True, sort=False)
        else:
            return pattern

    def colocalize_pattern(
            self, fixed_name, n_colocalize, colocalize_name=None,
            mode=None, fixed_fraction=1, colocalize_fraction=1,
            region=None, region_id=1, region_coords=None, max_dist=0,
            class_col_read=None, class_col_write=None, coord_cols=None,
            shuffle=True, keep_fixed=False, rng=None, seed=None, update=False):
        """Generates a colocalized (interacting) point pattern.

        Particles are defined to interact if the distance between
        them is at most interact_d.

        Essentially a wrapper around point_pattern.colocalize_pattern():
          - Extracts "fixed" particle set from self.particles table
          by selecting particles that have (arg) fixed_name in
          (arg) class_name_read column
          - Uses point_pattern.colocalize_pattern() to generate
          a pattern (particle coordinates) that is colocalized with the
          fixed pattern. Most arguments are simply passed (see
          colocalize_pattern() doc for more info)
          - Optionally adds the geneated set to self.particles table.
          Generated coordinates are saved in (arg) coord_cols and the
          value of (arg) colocalize_name is written in (possibly
          multiple) olumns specified by (arg) class_name_write. Values
          written to other column are copied from the first particle
          of the fixed particle pattern.

        If mode is 'max1', at most one point is colocalized with each
        selected fixed pattern point.

        If mode is 'kd', colocalize fraction is calculated from the
        equilibrium condition as:
          fixed_fraction * fixed_pattern.shape[0] / n_colocalize
        and arg fixed_fraction is ignored. 

        If mode is None, or 'many_to1', multiple points can be colocalized
        with any of the selected fixed points. The actual number of
        colocalized points is stochastic with expectation
        n_colocalize * colocalize_fraction.

        Important: In order to use the pattern generated here to
        calculate colocalization (by ColocLite.colocalize()) the
        name of the new pattern has to be written to self.subclass_col
        column of self.particles table.

        The generated points are ordered so that the points at the
        beginning are colocalized (expected number is
        n_colocalize * colocalized_fraction, but the actual number is
        stochastic) and the remaining points are not colocalized.
        
         Arguments:
          - fixed_name: name of the fixed pattern
          - n_colocalize: number of colocalized points to generate
          - colocalize_name: name of the colocalized (generated) pattern
          - mode: defines how are points colocalized with the fixed pattern,
          'many_to1' (same as None, default), 'max1' or 'kd'
          - class_col_read: column that contains fixed pattern name, 
          self.subclass_col is used if None (default)
          - class_col_write: one or more columns where the generated
          patter name is written, [self.class_name_col, self.subclass_col]
          is used if None (default)
          - coord_cols: column names containing pattern coordinates,
          self.coord_reg_frame_cols is used if None (default)
          - fixed_fraction: fraction of the fixed points used for
          colocalization (default 1)
          - colocalized_fraction: fraction of the colocalized points that are
          colocalized (default 1)
          - region: (ndarray or pyto.core.Image) label image (default None)
          - region_id: id of the region in the label image (region)
          (default 1)
          - region_coords: (ndarray, n_region_coords x n_dim) coordinates
          of all points of a region, used if args region and region_id
          are not specified (default None)
          - max_dist: distance (in pixels) that defines colocalization
          (interaction) neighborhoods (default 0)
          - shuffle: flag indicating if points are randomly shuffled before
          a fraction is selected (strongly recommended, default True)
          - rng: random number generator for shuffling, if None a new one
          is created
          - seed: random number generator seed for shuffling, used in case
          a new random number generator is created here

        Returns: (ndarray n_colocalize x n_dim) generated pattern if
        update is False.

        Updates self.particles table if update is True
        """

        # default arguments
        if class_col_read is None:
            class_col_read = self.subclass_col
        if class_col_write is None:
            class_col_write = [self.class_name_col, self.subclass_col]
        if isinstance(class_col_write, str):
            class_col_write = [class_col_write] 
        if coord_cols is None:
            coord_cols = self.coord_reg_frame_cols
        if rng is None:
            rng = default_rng(seed=seed)
        
        # read and split fixed pattern
        fixed_tab = self.particles[self.particles[class_col_read] == fixed_name]
        fixed = fixed_tab[coord_cols].to_numpy()

        # generate colocalized
        shuffle_fixed = shuffle and not keep_fixed
        colocalized = point_pattern.colocalize_pattern(
            fixed_pattern=fixed, n_colocalize=n_colocalize,
            mode=mode, fixed_fraction=fixed_fraction,
            colocalize_fraction=colocalize_fraction,
            region=region, region_id=region_id, region_coords=region_coords,
            max_dist=max_dist,
            shuffle_fixed=shuffle_fixed, shuffle_region=shuffle,
            rng=rng, seed=seed)

        # save or return
        if update:

            # make colocalized table
            colocalize_local = fixed_tab.iloc[0].to_dict()
            for ccol in class_col_write:
                colocalize_local[ccol] = colocalize_name
            for ax_ind, co_col in enumerate(coord_cols):
                colocalize_local[co_col] = colocalized[:, ax_ind]
            colocalize_local_tab = pd.DataFrame(colocalize_local)

            # add to particles tab
            self.particles = pd.concat(
                [self.particles, colocalize_local_tab],
                ignore_index=True, sort=False)
            
        else:
            
            return colocalized
