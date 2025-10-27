"""
One colocalization for one tomo, both real and simulated particles.

Contains class ColocOne.
 
# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"

import numpy as np
from numpy.random import default_rng
import scipy as sp
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.ndimage import distance_transform_edt
import pandas as pd 

from . import point_pattern
from . import coloc_functions as col_func
from .coloc_core import ColocCore


class ColocOne(ColocCore):
    """Class for one colocalization and one tomo, both real and simulated 
    particles.

    Colocalization is calculate using make_one() method.

    Important attributes
    """

    def __init__(
            self, full_coloc=True, keep_dist=False, mode='less', pixel_nm=1,
            metric='euclidean', columns=True, column_factor=2,
            name_mode='_', prefix='pattern', suffix='data',
            n_columns_mode='dist', p_func=np.greater,
            dist_mode='fine', seed=None, rng=None, max_iter=100, n_factor=2,
            all_random=False):
        """
        Sets attributes

        Arguments:
          - full_coloc: if True (default), when 3-colocalization is
          calculated, also the 2-colocalizations between the first pattern
          and each of the other patterns is also calculated  
          - columns: Flag 
        """

        super().__init__(
            full_coloc=full_coloc, keep_dist=keep_dist, mode=mode,
            pixel_nm=pixel_nm, metric=metric, columns=columns,
            column_factor=column_factor, prefix=prefix, suffix=suffix,
            n_columns_mode=n_columns_mode)

        # set from arguments
        #self.simul_suffix = simul_suffix
        self.name_mode = name_mode
        self.p_func = p_func
        self.dist_mode = dist_mode
        self.max_iter = max_iter
        self.n_factor = n_factor
        if rng is None:
            rng = default_rng(seed=seed)
        self.rng = rng
        self.seed = seed

        # default values of simulation related table column names
        self.set_simulation_suffixes(all_random=all_random)

    def set_simulation_suffixes(self, all_random=False):
        """Set attributes defining random simulation variable suffixes.

        Sets the following attributes if all_random is False (default):
          - random_suffixes: ['random', 'random_alt', 'random_combined']
          - simul_suffixes: ['simul_normal', 'simul_other']
          - p_suffixes: ['normal', 'other', 'combined']

        if all_random is True:
          - random_suffixes: ['random']
          - simul_suffixes: ['simul']
          - p_suffixes: ['solo']

        and in both cases:
          - simul_index_col: 'simul_id'

        Argument:
          - all_random: False (default) for the standard two simulation types 
          (called 'normal' and 'other'), and True for simulations where
          all sets are randomized simultaneously
        """

        self.all_random = all_random

        # should be the same as in ColocLite
        if not all_random:

            # default, two simulation types
            self.random_suffixes = ['random', 'random_alt', 'random_combined']
            self.simul_suffixes = ['simul_normal', 'simul_other']
            self.p_suffixes = ['normal', 'other', 'combined']
            self.simul_index_col = 'simul_id'

        else:

            # one simulation type, all random
            self.random_suffixes = ['random']
            self.simul_suffixes = ['simul']
            self.p_suffixes = ['solo']
            self.simul_index_col = 'simul_id'
        
    def make_one(
            self, patterns, regions, distance, n_simul, tomo_id='tomo',
            names=None, exclusion=None, exclusion_simul=None):
        """Colocalization for real and simulated particles for one tomo.

        Calculates colocalizations between the real point patterns 
        (individual components of arg patterns), and between the 
        corresponding random simulations. There are two types of random 
        simulations:
          - normal: between the first listed real pattern (patterns[0]) and
          the randomly generated patterns corresponding to the other real
          patterns
          - other: bewtween the randomly generated pattern corresponding
          to the first listed real pattern and the other than the first
          real patterns (arg patterns[1:])

        Imposes particle exclusion distance, see dics in point_particles module,
        functions random_region() and point_exclusion().

        Sets attributes:
          - patterns_clean (n_patterns, n_points x n_dims): point patterns
          (particle set) obtained by applying exclusion on (arg) patterns
          - Attributes corresponding to the calculated colocalization cases.
          For example, if names=['patternX', 'patternY', 'patternZ'], the
          attributes are:
            - self.patternX_patternY_patternZ+suffix: 3-colocalization
            - self.patternX_patternY+suffix: 2-colocalization
            - self.patternX_patternZ+suffix: 2-colocalization
          - self.data_names
          - self.bare_multid: (dict) BareColoc objects (values) for 
          distances (keys)

        Arguments:
          - patterns: list of point patterns (length n_patterns), where each
          pattern is an ndarray of shape (n_points, n_dim) containing point 
          pattern coordinates in pixels (thus ints)
          - regions: (one or a list of ndarrays) Label images that define
          regions where point patterns are located
          - distance: (list) colocalization distances in nm
          - n_simul: number of simulations
          - names: (list) point pattern names, has to correspond directly 
          to arg patterns, if None set to
          [self.prefix+'0', self.prefix+'1', self.prefix+'2']  
          - exclusion: exclusion distance for real patterns in nm
          - exclusion_simul: exclusion distance for simulated patterns in nm

        Region
        """
        
        n_patterns = len(patterns)
        if names is None:
            names = [f'{self.prefix}{i}' for i in range(n_patterns)]
        if not isinstance(regions, (list, tuple)):
            regions = n_patterns * [regions]
        
        # impose exclusion on real patterns and count the remaining points
        if exclusion is not None:
            exclusion_pix = exclusion / self.pixel_nm
            patterns_clean = [
                point_pattern.exclude(
                    points=pat, exclusion=exclusion_pix, metric=self.metric,
                    mode=self.dist_mode)
                for pat in patterns]
        else:
            patterns_clean = patterns
        self.patterns_clean = patterns_clean
        n_points = []
        for pat in patterns_clean:
            if (pat is None) or (pat.size == 0):
                n_points.append(0)
            else:
                n_points.append(pat.shape[0])

        # real coloc
        self.region = regions[0]
        self.make(
            patterns=patterns_clean, distance=distance, tomo_id=tomo_id,
            names=names, region=regions[0], suffix=self.suffix)

        # make and shuffle region coords 
        region_coords = [np.stack(np.nonzero(reg), axis=1) for reg in regions]
        for ind in range(len(regions)):
            self.rng.shuffle(region_coords[ind])

        # make instances for normal and other random simulations
        simul_normal = self.__class__(
            full_coloc=self.full_coloc, keep_dist=False, mode=self.mode,
            pixel_nm=self.pixel_nm, metric=self.metric, columns=False,
            column_factor=self.column_factor, prefix=self.prefix,
            n_columns_mode=self.n_columns_mode, all_random=self.all_random) 
        simul_other = self.__class__(
            full_coloc=self.full_coloc, keep_dist=False, mode=self.mode,
            pixel_nm=self.pixel_nm, metric=self.metric, columns=False,
            column_factor=self.column_factor, prefix=self.prefix,
            n_columns_mode=self.n_columns_mode, all_random=self.all_random)

        # simulate
        if exclusion_simul is not None:
            exclusion_simul_pix = exclusion_simul / self.pixel_nm
        else:
            exclusion_simul_pix = None
        for sim_ind in range(n_simul):

            if not self.all_random:
            
                # standard simulation
                simul_coords_normal = [
                    point_pattern.random_region(
                        N=n_pnt, region_coords=reg_coords,
                        exclusion=exclusion_simul_pix, metric=self.metric,
                        mode=self.dist_mode, shuffle=False, rng=self.rng,
                        max_iter=self.max_iter, n_factor=self.n_factor)
                    for n_pnt, reg_coords
                    in zip(n_points[1:], region_coords[1:])]
                patterns_normal = [patterns_clean[0]] + simul_coords_normal 
                simul_normal.make(
                    patterns=patterns_normal, distance=distance,
                    tomo_id=tomo_id, names=names, region=regions[0],
                    suffix=self.simul_suffixes[0])
                self.add_coloc(
                    coloc=simul_normal, extra={self.simul_index_col: sim_ind})  

                # other simulation
                simul_coords_other = point_pattern.random_region(
                    N=n_points[0], region_coords=region_coords[0],
                    exclusion=exclusion_simul_pix,
                    metric=self.metric, mode=self.dist_mode, shuffle=False,
                    rng=self.rng, max_iter=self.max_iter,
                    n_factor=self.n_factor)
                patterns_other = [simul_coords_other] + patterns_clean[1:]
                simul_other.make(
                    patterns=patterns_other, distance=distance,
                    tomo_id=tomo_id, names=names, region=regions[0],
                    suffix=self.simul_suffixes[1])
                self.add_coloc(
                    coloc=simul_other, extra={self.simul_index_col: sim_ind})  

            else:
                
                # all random simulation
                simul_coords_all_rand = [
                    point_pattern.random_region(
                        N=n_pnt, region_coords=reg_coords,
                        exclusion=exclusion_simul_pix, metric=self.metric,
                        mode=self.dist_mode, shuffle=False, rng=self.rng,
                        max_iter=self.max_iter, n_factor=self.n_factor)
                    for n_pnt, reg_coords
                    in zip(n_points, region_coords)]
                patterns_normal = simul_coords_all_rand
                simul_normal.make(
                    patterns=patterns_normal, distance=distance,
                    tomo_id=tomo_id, names=names, region=regions[0],
                    suffix=self.simul_suffixes[0])
                self.add_coloc(
                    coloc=simul_normal, extra={self.simul_index_col: sim_ind})  
                
        # combine simulation data with real
        self.combine_simulations()
        #print(self.X_Y_Z_data.shape)
        #print(self.X_Y_Z_data.columns)

        # p-values
        if self.full_coloc:
            coloc_names = col_func.make_full_coloc_names(
                names=names, suffix=self.suffix)
        else:
            coloc_names = [col_func.make_name(names=names, suffix=self.suffix)]
        for nam in coloc_names:
            data = getattr(self, nam)
            data_p = col_func.get_fraction_random(
                data=data, p_func=self.p_func, random_suff=self.random_suffixes,
                p_suff=self.p_suffixes)
            setattr(self, nam, data_p)
            
    def add_coloc(self, coloc, extra={}):
        """Add specified colocalization data to the current.
        
        Appends rows of colocalization results of another coloc instance 
        (arg coloc) to the corresponding tables of this instance 
        (corresponding means having the same name).

        If the current instance does not contain an attribute of the same 
        name as one of the colocalization results attributes of the 
        specified instance (arg coloc), the results table is set as a new
        attribute of this instance.

        Before rows are appended, items specified by arg extra are added as
        new columns to all the data of coloc.

        Arguments:
          - coloc: (ColocCore) colocalization object
          - extra: (dict) keys and values are interpreted as column names 
          and values, and added to coloc  
        """

        # loop over coloc data table names
        for data_nam in coloc.data_names:

            # add extra info
            new = getattr(coloc, data_nam)
            for key, value in extra.items():
                new[key] = value

            # add to existing
            try:
                current = getattr(self, data_nam)
            except (AttributeError, NameError):
                current = getattr(coloc, data_nam)
                if data_nam not in self.data_names:
                    self.data_names.append(data_nam)
            else:
                current = pd.concat([current, new], ignore_index=True)
            setattr(self, data_nam, current)

    def combine_simulations(self):
        """Combines simulation data with the real (experimental) data.

        """

        # loop over real colocalization results
        for data_nam in self.data_names:
            if not data_nam.endswith('_' + self.suffix):
                continue
            coloc_name = data_nam.removesuffix('_' + self.suffix)

            # get real and simulation data tables
            tab = getattr(self, data_nam)
            tab_simul_normal = getattr(
                self, coloc_name + '_' + self.simul_suffixes[0])
            if not self.all_random:
                tab_simul_other = getattr(
                    self, coloc_name + '_' + self.simul_suffixes[1])
            else:
                tab_simul_other = None
                
            # n subcolumns
            tab = self.combine_simulations_one(
                tab=tab, tab_simul_normal=tab_simul_normal,
                tab_simul_other=tab_simul_other, column='n_subcol',
                stats=True)

            # n particles in subcolumns
            for pattern_name in col_func.get_names(
                    name=coloc_name, mode=self.name_mode):
                tab = self.combine_simulations_one(
                    tab=tab, tab_simul_normal=tab_simul_normal,
                    tab_simul_other=tab_simul_other,
                    column=f'n_{pattern_name}_subcol', stats=False)
                
            # set the resulting table back to the attribute
            setattr(self, data_nam, tab)    
            
    def combine_simulations_one(
            self, tab, tab_simul_normal, tab_simul_other, column, stats=False):
        """Add simulations data to the main table for one feature.

        """
            
        # group data and put all values in single column
        grouped_normal = tab_simul_normal.groupby('distance')[column]
        df_normal_all = (
            grouped_normal
            .apply(lambda x: x.values)
            .to_frame(name=f'{column}_{self.random_suffixes[0]}_all'))
        if not self.all_random:
            grouped_other = tab_simul_other.groupby('distance')[column]
            df_other_all = (
                grouped_other
                .apply(lambda x: x.values)
                .to_frame(name=f'{column}_{self.random_suffixes[1]}_all'))

        if not stats:
            if not self.all_random:
                calculated = pd.concat([df_normal_all, df_other_all], axis=1)
            else:
                calculated = df_normal_all
            tab = pd.merge(
                tab, calculated, left_on='distance', right_index=True,
                sort=False)
            return tab            

        # calculate mean and std for normal
        normal_mean_col = f'{column}_{self.random_suffixes[0]}_mean'
        df_normal_mean = (
            grouped_normal
            .mean()
            .to_frame(name=normal_mean_col))
        normal_std_col = f'{column}_{self.random_suffixes[0]}_std'
        df_normal_std = (
            grouped_normal
            .std()
            .to_frame(name=normal_std_col))
        df_normal_count = (
            grouped_normal
            .count()
            .to_frame(name='count'))

        # calculate mean and std for other
        if not self.all_random:
            other_mean_col = f'{column}_{self.random_suffixes[1]}_mean'           
            df_other_mean = (
                grouped_other
                .mean()
                .to_frame(name=other_mean_col))
            other_std_col = f'{column}_{self.random_suffixes[1]}_std'           
            df_other_std = (
                grouped_other
                .std()
                .to_frame(name=other_std_col))
            df_other_count = (
                grouped_other
                .count()
                .to_frame(name='count'))

        # add means and stds to the table
        if not self.all_random:
            calculated = pd.concat(
                [df_normal_all, df_other_all,
                 df_normal_mean, df_normal_std, df_other_mean, df_other_std],
                axis=1)
        else:
             calculated = pd.concat(
                [df_normal_all, df_normal_mean, df_normal_std], axis=1)           
        tab = pd.merge(
            tab, calculated, left_on='distance', right_index=True,
            sort=False)

        if not self.all_random:

            # calculate combined mean and std
            combined_count = df_normal_count['count'] + df_other_count['count']
            combined_mean = (
                (df_normal_mean[normal_mean_col] * df_normal_count['count']
                 + df_other_mean[other_mean_col] * df_other_count['count'])
                / combined_count)
            sumsq_normal = (
                df_normal_std[normal_std_col]**2
                * (df_normal_count['count'] - 1)
                + df_normal_count['count'] * df_normal_mean[normal_mean_col]**2)
            sumsq_other = (
                df_other_std[other_std_col]**2 * (df_other_count['count'] - 1)
                +  df_other_count['count'] * df_other_mean[other_mean_col]**2)
            combined_var = (
                (sumsq_normal + sumsq_other - combined_count * combined_mean**2)
                / (df_normal_count['count'] + df_other_count['count'] - 1))
            combined_std = combined_var.map(lambda x: np.sqrt(x))        

            # add combined mean and std to table
            combined_mean_col = f'{column}_{self.random_suffixes[2]}_mean'
            tab[combined_mean_col] = combined_mean.values
            combined_std_col = f'{column}_{self.random_suffixes[2]}_std'
            tab[combined_std_col] = combined_std.values

        return tab



