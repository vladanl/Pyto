"""
Interface between particle sets (MultiParticleSets) and line projection
(LineProjection).

# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"


import numpy as np
import scipy as sp
import pandas as pd

import pyto
from .point_pattern import get_region_coords
from pyto.spatial.line_projection import LineProjection


class MPSProjection:
    """Projecting particles along membrane normals.

    """

    def __init__(
            self, line_spreads=[1], linep_cols_base='linep',
            combined_cols_base='combp',
            not_found=[-1, -1, -1], region_id=1,
            normal_angle_cols=['rlnAngleTilt', 'rlnAnglePsi'],
            total_col='total', unique_col_base='unique', same_col_base='same',
            missed_col_base='missed', unique_comb_col_base='unique_comb',
            same_comb_col_base='same_comb',
            best_col='best_spread', maximize_cols_base='unique_comb'):
        """Sets attributes from arguments.
        """

        self.line_spreads = line_spreads
        self.linep_cols_base = linep_cols_base
        self.combined_cols_base = combined_cols_base
        self.not_found = not_found
        self.region_id = region_id
        self.normal_angle_cols = normal_angle_cols

        # analysis tanle colums
        self.total_col = total_col
        self.unique_col_base = unique_col_base
        self.missed_col_base = missed_col_base
        self.same_col_base = same_col_base
        self.unique_comb_col_base = unique_comb_col_base
        self.same_comb_col_base = same_comb_col_base
        self.best_col = best_col
        self.maximize_cols_base = maximize_cols_base

    def project_multi(self, mps, coord_cols, distance, reverse=False):
        """Projects particles along their normals for different widths.

        """

        for grid_mode in self.line_spreads:

            line_project_cols = [
                f'{axis}_{self.linep_cols_base}_{grid_mode}'
                for axis in ['x', 'y', 'z']]
            dist_col = f'{self.linep_cols_base}_dist_{grid_mode}'

            # make projections 
            self.project(
                mps=mps, coord_cols=coord_cols, out_cols=line_project_cols,
                distance=distance, reverse=reverse, grid_mode=grid_mode)

            # calculate projection length
            mps.short_projected(
                non_projected_cols=coord_cols,
                projected_cols=line_project_cols,
                project_dist_col=dist_col)

    def project(
            self, mps, coord_cols, out_cols,
            distance, reverse=False, grid_mode=1, region_col=None):
        """Projects particles along their normals for multiple tomos.

        If for a particle a projection is not found, coordinates specified
        by arg not_found are entered. This should be something like
        [-1, -1, -1] (default). Alternatively, if not_found is 
        [NaN, NaN, NaN], coord_cols are converted to 'Int64', which is
        different from the standard 'int'.

        """

        # set default
        if region_col is None:
            region_col = mps.region_col
        
        part_list = []
        for _, tomo_row in mps.tomos.iterrows():

            # get tomo data
            tomo_id = tomo_row[mps.tomo_id_col]
            region_path = tomo_row[region_col]

            # project
            part_one = mps.particles[mps.particles[
                mps.tomo_id_col]==tomo_id].copy()
            self.project_one(
                particles=part_one, region=region_path,
                region_id=self.region_id, coord_cols=coord_cols,
                angle_cols=self.normal_angle_cols, out_cols=out_cols,
                distance=distance, reverse=reverse, grid_mode=grid_mode,
                not_found=self.not_found)
            part_list.append(part_one)

        mps.particles = pd.concat(part_list)
        
    def project_one(
            self, particles, region, region_id, coord_cols, angle_cols,
            out_cols, distance, reverse=False, grid_mode=1,
            not_found=[-1,-1,-1]):
        """Projects MPS particles along their normals for one tomo.

        """

        # figure out region points
        if isinstance(region, str):
            region = pyto.segmentation.Labels.read(
                file=region, memmap=True)
            region_coords = get_region_coords(
                region=region, region_id=region_id, shuffle=False)

        # find projections
        points = particles[coord_cols].to_numpy()
        angles = particles[angle_cols].to_numpy()
        n_points = particles.shape[0]
        lp = LineProjection(
            region_coords=region_coords, relion=True, reverse=reverse,
            grid_mode=grid_mode, intersect_mode='first', not_found=not_found)
        particles[out_cols] = particles.apply(
            lambda x: lp.project(
                point=x[coord_cols].to_numpy(),
                angles=x[angle_cols].to_numpy(), distance=distance),
            axis=1, result_type='expand')
        
        # add to particles
        if particles[out_cols].isna().any(axis=None):
            particles[out_cols] = particles[out_cols].astype('Int64')

    def find_best_projections(self, mps, distance_cols=None):
        """Finds the best projection line width for each tomo.
    
        Combines normal line and distance based projections (see
        combine_projections()) for multiple line widths, analyses 
        them (see analyze_projections()) and finds the optimal line 
        width. 

        The optimal line width (grid mode) results in the highest 
        number of uniqe projections.
    
        """

        if distance_cols is None:
            distance_cols = mps.coord_reg_frame_cols
    
        project_analysis = pd.DataFrame()
        for grid_m in self.line_spreads:

            line_project_cols = [
                f'{axis}_{self.linep_cols_base}_{grid_m}'
                for axis in ['x', 'y', 'z']]
            combine_project_cols = [
                f'{axis}_{self.combined_cols_base}_{grid_m}'
                for axis in ['x', 'y', 'z']]

            self.combine_projections(
                mps=mps, distance_cols=distance_cols,
                line_cols=line_project_cols, combine_cols=combine_project_cols)

            analysis_one = self.analyze_projections(
                mps=mps, distance_cols=distance_cols,
                line_cols=line_project_cols, combine_cols=combine_project_cols,
                grid_mode=grid_m)
    
            # merge current analysis 
            try:
                project_analysis = project_analysis.merge(
                    analysis_one, on=[mps.tomo_id_col, self.total_col],
                    sort=False)
            except KeyError:
                project_analysis = analysis_one

        # find the best 
        project_analysis = self.select_projections(
            mps=mps, analysis=project_analysis)
            
        return project_analysis
            
    def combine_projections(
            self, mps, distance_cols, line_cols, combine_cols=None):
        """Combines line and distance based projections.
    
        Takes values from line projections (columns line_cols) and replaces 
        those where projections could not be made (having value not_found)
        by minimum distance based projections (columns distance cols).
        Writes these values in combined columns (name combine_cols). 
    
        Adds column combine_cols to self.particles. 
        """
    
        if combine_cols is None:
            combine_cols = line_cols
        no_line = (mps.particles[line_cols] == self.not_found).all(axis=1)
        mps.particles[combine_cols] = np.where(
            no_line.to_numpy().reshape(-1, 1), 
            mps.particles[distance_cols].to_numpy(),
            mps.particles[line_cols].to_numpy())
    
    def analyze_projections(
            self, mps, distance_cols, line_cols, combine_cols, grid_mode):
        """Analyzes combining distance and line projections.
    
        """
    
        mps_groups = mps.particles.groupby(mps.tomo_id_col)
        tomo_ids = []
        n_total = []
        n_unique = []
        n_missed = []
        n_unique_after = []
        for to_id, particles_one in mps_groups:
            tomo_ids.append(to_id)
            n_total.append(particles_one.shape[0])
        
            # line projection analysis
            mis = np.logical_and.reduce(
                particles_one[line_cols] == self.not_found, axis=1).sum()
            uni = np.unique(particles_one[line_cols].to_numpy(), axis=0).shape[0]
            if mis > 0:
                uni = uni - 1
            n_unique.append(uni)
            n_missed.append(mis)
        
            # combined analysis
            uni_after = np.unique(
                particles_one[combine_cols].to_numpy(), axis=0).shape[0]
            n_unique_after.append(uni_after)
        
        # put in dataframe
        n_same = np.asarray(n_total) - np.asarray(n_missed) - np.asarray(n_unique)
        n_same_after = np.asarray(n_total) - np.asarray(n_unique_after)
        res = pd.DataFrame(
            {mps.tomo_id_col: tomo_ids, self.total_col: n_total, 
             f'{self.unique_col_base}_{grid_mode}': n_unique,
             f'{self.missed_col_base}_{grid_mode}': n_missed, 
             f'{self.same_col_base}_{grid_mode}': n_same,
             f'{self.unique_comb_col_base}_{grid_mode}': n_unique_after,
             f'{self.same_comb_col_base}_{grid_mode}': n_same_after})
    
        return res
     
    def select_projections(self, mps, analysis):
        """Select projections from the best grids mode for each tomo.
    
        
        """
    
        # find best grid mode
        maximize_cols = [
            f"{self.maximize_cols_base}_{grid_m}"
            for grid_m in self.line_spreads]
        analysis[self.best_col] = (
            analysis[maximize_cols]
            .idxmax(axis=1)
            .apply(lambda x: int(
                x.removeprefix(self.maximize_cols_base+'_'))))
    
        # extract best projections
        combined_all = []
        for _, tomo_row in mps.tomos.iterrows():
            tomo_id = tomo_row[mps.tomo_id_col]
            analysis_row = analysis[analysis[mps.tomo_id_col] == tomo_id]
            parts_one = mps.particles[mps.particles[mps.tomo_id_col] == tomo_id]

            grid_m = analysis_row[self.best_col].to_numpy()[0]
            comb_best_cols = [
                f'{axis}_{self.combined_cols_base}_{grid_m}'
                for axis in ['x', 'y', 'z']]
            comb_final_cols = [
                f'{axis}_{self.combined_cols_base}'
                for axis in ['x', 'y', 'z']]
            combined_one = parts_one[comb_best_cols].copy()
            col_rename = dict(zip(comb_best_cols, comb_final_cols))
            combined_one.rename(columns=col_rename, inplace=True)
            combined_all.append(combined_one)
        
        # put best projections together and add to particles
        combined = pd.concat(combined_all, axis=0)
        mps.particles = mps.particles.join(combined, how='left')
    
        return analysis

            
