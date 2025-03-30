"""
Methods that provide some functionality to MultiParticleSets

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

import abc
import os
import itertools

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

import pyto
from pyto.util.pandas_plus import merge_left_keep_index
from pyto.segmentation.labels import Labels
import pyto.spatial.coloc_functions as col_func
from .point_pattern import get_region_coords, exclude, project


class MPSAnalysis(abc.ABC):
    """Abstract class that provides MPS analysis methods.

    Meant to be inherited by MultiParticleSets.
    """

    @abc.abstractmethod
    def __init__(self):
        pass

    def find_min_distances(
            self, df_1, df_2=None, group_col=None, coord_cols_1=None,
            coord_cols_2=None, distance_col='distance', ind_col_2='index_2'):
        """Finds the closest particle of df_2 for each element of df_1.

        If arg df_2 is None, finds the closest particle within df_1.

        The closest particle is searched within subgroups defined by arg 
        group_col.

        If the closest element cannot be determined, both the distance
        and the index in the resulting table are set to -1. This can 
        happen because there are no particles in the corresponding group 
        of df_2, or df_2 is None and a group contains only one particle)

        Arguments:
          - df_1, df_2: (pandas.DataFrames) tables containing particle 
          coordinates
          - group_col: (str, or list of tuple of strs) One or more Column 
          names that distinguishes different groups within tables df_1 and 
          df_2, such as 'tomo_id' (has to be the same in both tables)
          - coord_cols_1, coord_cols_2: Names of columns containing coordinates
          in df_1 and df_2; if coord_cols2 is None it is set to coord_cols_1
          - distance_col: name of the column containing distance in the 
          resulting table
          - distance_col: name of the column containing index values of df_2 
          in the resulting table

        Returns pandas.DataFrame containing smallest distances, with columns:
          - index: the same as df_1.index
          - group_col: group name
          - coord_cols_1: coordinates of df_1
          - ind_col_2: indices of df_2 that point to the closest particles of
          df_1 (one for each df_1 particle)
          - distance_col: distances between df_1 particles and their closest
          df_2 particles
        """

        # default args
        if coord_cols_1 is None:
            coord_cols_1 = self.coord_reg_frame_cols
        if coord_cols_2 is None:
            coord_cols_2 = coord_cols_1

        # check if df-1 and df_2 the same
        df_same = False
        if df_2 is None:
            df_2 = df_1
            df_same = True

        # check if multiple groups
        group_multi = False
        if isinstance(group_col, (list, tuple)):
            if len(group_col) > 1:
                group_multi = True
                group_col_list = group_col
            else:
                group_col = group_col[0]
                group_col_list = [group_col]
        else:
            group_col_list = [group_col]
        
        # separate df-1 and df_2 into groups 
        if group_col is None:
            # allow a single group wo selection
            df = df_1[coord_cols_1].copy()
            iter_list = [(None, df_1, df_2)]
            
        else:
            df = df_1[group_col_list + coord_cols_1].copy()
            groups = df.groupby(group_col)
            if not group_multi:

                # single group selection
                iter_list = [
                    (name, groups.get_group(name),
                     df_2[df_2[group_col] == name])
                    for name in groups.groups.keys()]

            else:

                # multiple group selection
                iter_list = []
                for name in groups.groups.keys():
                    cond_2_list = [
                        df_2[group_col_one] == name_one
                        for group_col_one, name_one in zip(group_col, name)]
                    cond_2 = np.logical_and.reduce(cond_2_list)
                    iter_list.append(
                        (name, groups.get_group(name), df_2[cond_2]))
            
        # loop over df_1 particles (rows)
        for name, group_1, group_2 in iter_list:

            # get coords
            #group_2 = df_2[df_2[group_col] == name]
            coords_1 = group_1[coord_cols_1].values
            coords_2 = group_2[coord_cols_2].values

            # find min distanes and indices
            if coords_2.shape[0] == 0:
                closest_inds_2 = -1
                closest_dist = -1
            elif df_same and (coords_2.shape[0] == 1):
                closest_inds_2 = -1
                closest_dist = -1
            else:
                dist = cdist(coords_1, coords_2)
                if df_same:
                    dist_max = dist.max()
                    dist = dist + np.diag(coords_1.shape[0]*[dist_max+1])
                closest_inds_2_local = dist.argmin(axis=1)
                closest_inds_2 = group_2.iloc[closest_inds_2_local, :].index
                n_1 = coords_1.shape[0]
                closest_dist = dist[range(n_1), closest_inds_2_local]
                #closest_dist_check = dist.min(axis=1)

            # add distances and indices 
            df.loc[group_1.index, ind_col_2] = closest_inds_2
            df.loc[group_1.index, distance_col] = closest_dist

        df = df.astype({ind_col_2: int})
        return df

    def near_neighbor_plot(
            self, particle_dfs, coord_cols, labels, 
            n_columns, figsize_one_row, tomos=None):
        """Plots nearest neighbor distributions for each tomo separatly.

        Args particle_dfs, coord_cols and labels have to have the same 
        number of elements that correspond to each other.

        """

        if tomos is None:
            tomos = np.sort(self.tomos[self.tomo_id_col].to_numpy())

        n_tomos = len(tomos)
        n_rows = np.ceil(n_tomos / n_columns).astype(int)

        fig, axes = plt.subplots(
            n_rows, n_columns, squeeze=False,
            figsize=(figsize_one_row[0], n_rows*figsize_one_row[1]))

        # loop over tomos
        for ind, to in enumerate(tomos):
            row = ind // n_columns
            column = np.mod(ind, n_columns)

            # loop over graphs
            for table, coord_co, lab in zip(particle_dfs, coord_cols, labels):
                coords = table[table['tomo_id'] == to][coord_co].to_numpy()
                if coords.size == 0:
                    continue
                nbrs = NearestNeighbors(n_neighbors=2).fit(coords)
                try:
                    distances, indices = nbrs.kneighbors(coords)
                except ValueError:
                    continue
                histo, bins = np.histogram(distances[:, 1], bins=np.arange(20))
                ax = axes[row, column]
                ax.plot(bins[:-1], histo, label=lab)

            ax.set_title(to)
            ax.legend()
        fig.tight_layout(w_pad=1.2)

        return fig, axes
    
    def in_region(
            self, tomos=None, particles=None, coord_cols=None,
            path_prefix=None):
        """Determines whether particles belong to the corresponding regions.

        Adds a new column (self.in_region_col) to self.particles table
        that indicates whether the particle is located inside the 
        corresponding tomo region. Takes projected particles by default,
        (column self.coord_reg_frame_cols), but this can be changed 
        using arg coord_cols.

        Particles that are determined to belong / not belog to regions 
        are labeled by True / False (column self.in_region_col), while 
        those that were not changed by numpy.nan.

        Tomo region paths and ids are read from self.tomos table.

        Modifies self.particles.

        Arguments:
          - tomos: list of tomos for which particles are checked, None for 
          all tomos
          - particles: (pandas.Dataframe) particles table, if None 
          self.particles is used
          - coord_cols: (list) particle coordinate columns
          - path_prefix: (str) prefix that is prepended to region image 
          paths obtained from self.tomos, in case these are relative, 
          needed in order to be able to call this method from any directory

        Return:
          - (pandas.Dataframe) particles table
        """

        if particles is None:
            particles = self.particles
        if coord_cols is None:
            coord_cols = self.coord_reg_frame_cols

        groups = particles.copy().groupby(self.tomo_id_col)
        for to, df_one in groups:
            
            # skip non-specified tomos
            if (tomos is not None) and (to not in tomos):
                continue

            # get regions image 
            this_tomo_cond = (self.tomos[self.tomo_id_col] == to)
            reg_path = self.tomos[this_tomo_cond][self.region_col].values[0]
            if (not os.path.isabs(reg_path)) and (path_prefix is not None):
                reg_path = os.path.normpath(os.path.join(path_prefix, reg_path))
            reg_id =  self.tomos[this_tomo_cond][self.region_id_col].values[0]
            reg_image = Labels.read(file=reg_path, memmap=True)

            # get region coords
            reg = reg_image.data
            reg_coords = np.stack(np.nonzero(reg==reg_id), axis=1)
            reg_coords_df = pd.DataFrame(reg_coords, columns=coord_cols)
            reg_coords_df[self.in_region_col] = True
        
            # determine whether in region
            coord_reg = (
                df_one
                .reset_index()
                .merge(reg_coords_df, left_on=coord_cols, right_on=coord_cols,
                       how='left', sort=False, suffixes=('_out', ''))
                .set_index('index'))

            # add in region to the table
            coord_reg[self.in_region_col] = \
                coord_reg[self.in_region_col].replace({np.nan: False})
            particles.loc[df_one.index, self.in_region_col] = \
                coord_reg[self.in_region_col]
            
        return particles

    def short_projected(
            self, project_dist_col, project_dist_max=None, 
            keep_project_dist_col=None,
            projected_cols=None, non_projected_cols=None):
        """Keep particles with short projection distance.

        Calculates projection distance, that is the distance between
        projected and non-projected particles (args projected_cols and
        non_projected_cols, respectively) for each particle. These
        distances are saved in (arg) project_dist_col.

        If arg project_dist_max is not None, makes a (boolean) Series that
        indicates whether the projection distances are <= than the given 
        value. 

        If arg keep_project_dist_col is not one of
        the current columns of self.particles, the above Series is added to
        self.particles as a new column named by keep_project_dist_col.

        Alternatively, if arg keep_project_dist_col already exists as a 
        self.particles column, this column is updated by subjecting its
        current valus to the and operation with the above Series values.

        Sets attributes:
          - self.particles: updated to include projection distances and
          possibly related flags
          - self.project_dist_col: arg project_dist_col

        Arguments:
          - project_dist_col: projection distances column name
          - project_dist_max: max projection distance (default None) 
          - keep_project_dist_col: keep projection distance flags column,
          if None (default) set to self.keep_col
          - projected_cols: projected particle coords columns, if None
          (default) set to self.coord_reg_frame_cols
          - non_projected_cols: before projection particle coord columns,
          if None (default) self.orig_coord_reg_frame_cols
        """

        # argument defaults
        if projected_cols is None:
            projected_cols = self.coord_reg_frame_cols
        if non_projected_cols is None:
            non_projected_cols = self.orig_coord_reg_frame_cols
        if keep_project_dist_col is None:
            keep_project_dist_col = self.keep_col

        # save arg value
        self.project_dist_col = project_dist_col

        # calculate projection distance
        parts = self.particles
        parts[self.project_dist_col] = np.linalg.norm(
            (parts[projected_cols].to_numpy()
             - parts[non_projected_cols].to_numpy()), axis=1)
        if project_dist_max is None:
            self.particles = parts
            return 
        
        # flag particles having long projection distance
        short_projected = parts[project_dist_col] <= project_dist_max
        if keep_project_dist_col not in parts.columns:
            parts[keep_project_dist_col] = short_projected
        else:
            parts[keep_project_dist_col] = (
                parts[keep_project_dist_col] & short_projected)
        self.particles = parts

    def extract_combination_sets_n(
            self, resolve_fn, set_names, subclass_col='subclass_name',
            pivot=True, group_col='group'):
        """Makes particle sets composed of one or more classes.

        Determines class name and class number(s) for each specified set name 
        (arg set_names) and makes an instance of this class that contains only
        those sets. Also adds a column to the particles table of the resulting
        instance that shows the set name. Particles table of the current 
        instance has to contain the particles that belong to the 
        specified sets.

        If arg pivot is True, makes a table containing number of particles for
        all groups (rows) and set name (columns).

        Arguments:
        - resolve_fn: function that determines the class name and class 
          numbers from a set name, with the folloing signature:
            _, class_name, list_of_class_numbers = resolve_fn(set_name)
        - set_names: (list) particle set names
        - subclass_col: name of the column that shows particle set names
        - pivot: flag indicationg whether to create the table containing
        article numbers
        - group_col: column name that contains properties used to group 
        particles

        Returns:
          new_instance, if pivot is False
          (new_instance, n_particle_table), if pivot is True
        where:
          - new_instance: instance of this class that contains only the 
          particles belonging to the specified sets
          - n_particle_table: table containing number of particles for
        all groups (rows) and set name (columns)
        """ 

        combined = self.__class__()
        combined.tomos = pd.DataFrame()
        combined.particles = pd.DataFrame()

        for s_nam in set_names:

            # select by current set (class)
            #_, cl_nam, cl_num = resolve_fn(set_name=s_nam)
            resolve_result = resolve_fn(set_name=s_nam)
            if isinstance(resolve_result, dict):
                one = self.select(rules=resolve_result, update=False)
            else:
                _, cl_nam, cl_num = resolve_result
                one = self.select(
                    class_names=[cl_nam], class_numbers=cl_num, update=False)
            one.particles[subclass_col] = s_nam

            # add current data
            combined.tomos = (
                pd.concat([combined.tomos, one.tomos])
                .drop_duplicates())
            combined.particles = pd.concat(
                [combined.particles, one.particles]).drop_duplicates()
            result = combined

        if pivot:
            comb_tab = combined.group_set_n(
                group_col=group_col, subclass_col=subclass_col)
            comb_tab = comb_tab[set_names]  # reorder
            result = (combined, comb_tab)

        return result

    def group_n(self, group_by, total=False):
        """Finds n particles for different combination of column values.

        Makes a table where row indices are values of arg
        group_by[0] and columns are values of group_by[1]. Values of this
        table are the number of particles for the corresponding 
        row/column combination.

        Table self.particles has to have rows other than those specified 
        in arg group_by.  

        Arguments:
          - group_by: (list, length 2) columns of self.particles  
          - total: flag indicating if column totals are added as the last row 

        Returns (pandas.DataFrame) number of particles.
        """

        index_col = group_by[0]
        other_group_cols = group_by[1:]
        try:
            other_col = [
                col for col in self.particles.columns if col not in group_by][0]
        except IndexError:
            raise ValueError(
                "Particles table has to have columns that are not specified "
                + f"in arg group_by {group_by}")
        try:
            comb_tab = (self.particles
                .groupby(
                    group_by, as_index=False)[other_col].count()
                .pivot(
                    index=index_col, columns=other_group_cols, values=other_col)
                .fillna(0)
                #.applymap(int)
                .map(int)
                )
        except AttributeError:
            # pandas version <2.1.0
            comb_tab = (self.particles
                .groupby(
                    group_by, as_index=False)[other_col].count()
                .pivot(
                    index=index_col, columns=other_group_cols, values=other_col)
                .fillna(0)
                .applymap(int)
                )

        if total:
            comb_tab.loc['Total'] = comb_tab.sum(axis=0)
        
        return comb_tab

    def group_set_n(self, group_col='group', subclass_col=None, total=False):
        """Finds n particles for all group / class combinations. 

        Based on self.group_n().

        Arguments:
          - group_col: column of self.paricles that contains groups
          - subclass_col: column of self particles that contain class names
          - total: flag indicating if column totals are added as the last row 

        Returns (pandas.DataFrame) number of particles, where rows are
        groups and columns are classes.
        """
        if subclass_col is None:
            subclass_col = self.class_number_col
        res = self.group_n(group_by=[group_col, subclass_col], total=total)
        return res

    @classmethod
    def update_star_classification(
            cls, in_path, out_path, star_paths, class_col, 
            class_label=None, class_value=None, 
            unique_cols=None, left_unique_cols=None, right_unique_cols=None,
            check=False, check_cols=None, right_check_cols=None,
            verbose=True):
        """Adds particle classifications from one or more star files.

        Not used in the momemnt (1.2024)
        To remove?
        """

        mps = cls.read(in_path, verbose=verbose)

        parts = mps.add_star_classification(
            star_paths=star_paths, class_col=class_col, 
            class_label=class_label, class_value=class_value, 
            unique_cols=unique_cols, left_unique_cols=left_unique_cols,
            right_unique_cols=right_unique_cols,
            update=False, check=check, check_cols=check_cols,
            right_check_cols=right_check_cols)

        mps.particles = parts
        mps.write(path=out_path, verbose=verbose)
        
    def add_star_classification(
            self, star_paths, class_col, particles=None,
            class_label=None, class_value=None, 
            unique_cols=None, left_unique_cols=None, right_unique_cols=None,
            update=False, check=False, check_cols=None, right_check_cols=None):
        """Adds particle classifications from one or more star files.

        Reads star files (arg star_paths) and adds classification info 
        to the corresponding particles of self.particles. 

        After reading, star files are converted to pandas.DataFrame and
        added to self.particles using self.add_classification(). See
        add_classification() docs for more info.

        Arguments:
          - star_paths: one or more star file paths
          - unique_cols, left_unique_cols and right_cols: names of the 
          columns of particles (left) and star files (right) that
          uniquely specify particles
          - class_col: column in self.particles to which the new classification
          is addedd
          - class_label: column of star files that holds the classification  
          - class_value: class name used if class_label is None
          - class_fmt: format of the class names added to self.particles 
          table (used only if class_label is not None) 
          - update: if True self.particles is updated, otherwise returns
          the modified particles table 
          - check: flag indicating if other column(s) should be checked
          to have the same values in particles and star file tables 
          - check_cols: columns that should be checked, used for both tables, 
          if right_check_cols is None, otherwise only for particles table
          - right_check_cols: columns of star files that should be 
          checked 
          - eps: numerical error limit, used to check if the corresponding 
          coordinates in particles and classification are equal (used only 
          if arg check is True)

        Returns: modified self.particles table if update is True, 
        nothing otherwise
        """

        # figure out particles
        if particles is None:
            particles = self.particles
            
        mps_star = self.__class__()

        for path, fmt in star_paths.items():
            
            parts_class = mps_star.read_star(path=path, mode='particle')
            particles = self.add_classification(
                particles=particles, classification=parts_class,
                class_col=class_col, class_label=class_label,
                class_value=class_value, class_fmt=fmt,
                unique_cols=unique_cols, left_unique_cols=left_unique_cols,
                right_unique_cols=right_unique_cols,
                update=False, check=check,
                check_cols=check_cols, right_check_cols=right_check_cols)       
        
        if update:
            self.particles = particles
        else:
            return particles

    def add_classification(
            self, classification, class_col, particles=None,
            class_label=None, class_value=None, class_fmt=None,
            unique_cols=None, left_unique_cols=None, right_unique_cols=None,
            update=False, check=False, check_cols=None, right_check_cols=None,
            eps=1e-9):
        """Adds classification to particles table.

        First, for each particle (row) of classification, the same particle
        in self.particles table is found, as follows:
          - if all three args unique_cols, left- and right_unique_cols are
          None, tomo id and particle id (columns self.tomo_id_col and 
          self.particle_id_col, respectively) are used for both particles and
          classification tables. This requers that the two columns taken 
          together uniquely specify a particle, as it is expected for
          colocalization by ColocLite
          - if arg unique_cols is specified, it is used for both tables
          - if arg unique_cols is None and left- and right_unique_cols 
          are specified, left_unique_cols of particles table have to match
          right_unique_cols of classification
          - if unique_cols, and at least one of left- and reight_unique_cols
          is specified, ValueError is raised

        Then, takes class names (values of the column class_label from 
        classification table and add them to column class_col of 
        self.particles table. If class_label is None, class_value is used 
        for all particles of classification table.

        If arg check is True, checks wheteher values of columns
        self.orig_coord_cols are the same for the corresponding particles 
        of tables self.particles and classification.

        Modifies arg classification.

        Arguments:
          - particles: particles table, if not specified self.particles
          is used
          - classification: (pandas.DataFrame) table that contains a 
          classification
          - unique_cols, left_unique_cols and right_cols: names of the 
          columns of particles (left) and classifications (right) that
          uniquely specify particles
          - class_col: column in self.particles to which the new classification
          is addedd
          - class_label: column in classification that holds the  
          classification table class names
          - class_value: class name used if class_label is None
          - class_fmt: format of the class names added to self.particles 
          table (used only if class_label is not None)
          - update: if True self.particles is updated, otherwise returns
          the modified particles table 
          - check: flag indicating if other column(s) should be checked
          to have the same values in particles and classification tables 
          - check_cols: columns that should be checked, used for both tables, 
          if right_check_cols is None, otherwise only for particles table
          - right_check_cols: columns of classification table that should be 
          checked 
          - eps: numerical error limit, used to check if the corresponding 
          coordinates in particles and classification are equal (used only 
          if arg check is True)

        Returns: modified self.particles table if update is True, 
        nothing otherwise
        """

        # arg defaults
        if particles is None:
            particles = self.particles
        if ((unique_cols is None) and (left_unique_cols is None)
            and (right_unique_cols is None)):
            unique_cols = [self.tomo_id_col, self.particle_id_col]
            left_unique_cols = None
            right_unique_cols = None
        elif unique_cols is not None:
            left_unique_cols = None
            right_unique_cols = None
            if ((left_unique_cols is not None)
                or (right_unique_cols is not None)):
                raise ValueError(
                    "Specifying both unique_cols and one of Left- and "
                    + "right_unique_columns is not allowed.")
        if check and (check_cols is None):
            check_cols = self.orig_coord_cols
            
        # setup new classification
        if class_label is not None:
            if class_fmt is not None:
                classification[class_col] = classification[
                    class_label].apply(lambda x: class_fmt.format(x))
            else:
                classification = classification.rename(
                    columns={class_label: class_col})
        else:
            classification[class_col] = class_value
        if unique_cols is None:
            right_cols = right_unique_cols + [class_col]
        else:
            right_cols = unique_cols + [class_col]
        if check:
            right_cols = right_cols + check_cols

        # add new classification
        parts = merge_left_keep_index(
            particles, classification[right_cols],
            on=unique_cols, left_on=left_unique_cols,
            right_on=right_unique_cols, suffixes=('', '_y'))

        # merge classification columns
        class_col_y = f'{class_col}_y'
        if class_col_y in parts.columns:
            parts[class_col] = np.where(
                ~parts[class_col_y].isnull(),
                parts[class_col_y], parts[class_col])
            parts.drop(columns=[class_col_y], inplace=True)

        # remove right unique columns that are different from the left
        if (left_unique_cols is not None) and (right_unique_cols is not None):
            differ = np.array(left_unique_cols) != np.array(right_unique_cols)
            to_remove = np.asarray(right_unique_cols)[differ]
            parts.drop(columns=to_remove, inplace=True)
            
        # check and clean
        if check:
            if right_check_cols is None:
                right_check_cols = [col + '_y' for col in check_cols]
            cond = np.logical_and.reduce(
                parts[right_check_cols].notnull().to_numpy(), axis=1)
            # need <eps because sometimes numerical error 
            check_passed = np.abs(np.logical_and.reduce(
                (parts.loc[cond, check_cols].to_numpy()
                 - parts.loc[cond, right_check_cols].to_numpy() < eps),
                axis=1))
            #check_passed = np.logical_and.reduce(
            #    (parts.loc[cond, check_cols].to_numpy()
            #     == parts.loc[cond, right_check_cols].to_numpy()),
            #    axis=1)
            if not check_passed.all():
                #raise ValueError(
                #    "Coordinates in particles and classification tables are "
                #    + f"different (in columns {check_cols})")
                print(
                    "WARNING: Check failed, coordinates in columns "
                    + f"{check_cols} in particles and classification tables "
                    + "are different. Columns from (arg) classification "
                    + "table are given suffix 'err'.")
                rename_check = dict(
                    (col, f'{col}_err') for col in right_check_cols)
                parts.rename(columns=rename_check, inplace=True)
            else:
                parts.drop(columns=right_check_cols, inplace=True)

        # add new classification column to self.classification_cols
        try:
            if class_col not in self.classification_cols:
                self.classification_cols.append(class_col)
        except AttributeError:
            self.classification_cols = [class_col]

        if update:
            self.particles = parts
        else:
            return parts

    def reclassify(
            self, classification, class_col, reclass_col, particles=None,
            update=False):
        """Reclassifies an already existing particle classification.

        Reclassifies the existing classes (of particles table) by combining
        into new classes according to arg classification.

        Arguments:
          - particles: particles table, if not specified self.particles
          is used
          - classification: (dict) keys are new class names and items
          are lists that contain the existing class names or numbers
          - class_col: name of the column that contains the existing 
          classification
          - reclass_col: name of the column that will contain the new 
          classification
          - update: if True self.particles is updated, otherwise returns
          the modified particles table 
          
        Returns: modified self.particles table if update is True, 
        nothing otherwise
         """

        if particles is None:
            particles = self.particles

        # add reclassification column to self.classification_cols
        try:
            if reclass_col not in self.classification_cols:
                self.classification_cols.append(reclass_col)
        except AttributeError:
            self.classification_cols = [reclass_col]

        # reclassify
        for reclass, classes in classification.items():
        
            cond_stack = np.array([
                (particles[class_col] == cl).to_numpy() for cl in classes])
            cond = np.logical_or.reduce(cond_stack, axis=0)
            particles.loc[cond, reclass_col] = reclass
            
        if update:
            self.particles = particles
        else:
            return particles

    def check_ids(
            self, expected, path_col=None, update=False,
            found_col='found_ids', verbose=True):
        """Checks if images contain expected ids.

        Arguments:
          - path_col: column of mps.particles that contain paths, if None
          (default) self.region_particle_col is used
        """
        
        if path_col is None:
            path_col = self.region_particle_col
        expected = np.asarray(expected)

        found_col_exists = False
        if found_col in self.particles.columns:
            found_col_exists = True

        # find particles that contain all segments
        all_found = True
        found_ids = []
        expected_set = set(expected)
        for pind, row in self.particles.iterrows():
            if found_col_exists and self.particles[found_col].to_numpy()[0]:
                found_ids.append(False)
                continue
            path = row[path_col]
            image = pyto.segmentation.Labels.read(path)
            actual = image.extractIds()
            this_found = expected_set == set(actual)
            #try:
            #    this_found = this_found.all()
            #except AttributeError:
            #    this_found = False
            if not this_found:
                all_found = False
                if verbose:
                    print(f"Particle {path}: Ids found: {actual}, "
                          + f"expected: {expected} ")
            found_ids.append(this_found)

        # update table
        if update:
            if found_col_exists:
                self.particles[found_col] = (
                    self.particles[found_col] & found_ids)
            else:
                self.particles[found_col] = found_ids
            
        if all_found and verbose:
            print("All particles have expected ids")

    #
    # Methods that select or group particles
    #
    
    def select(
            self, tomo_ids=None, class_names=None, class_numbers=None,
            rules=None, update=True):
        """Selects tomos and particles from the current instance.

        Selects tomos and particles that have tomo ids (column 
        self.tomo_id_col), class names (column self.class_name_col)
        and class numbers (column self.class_number_col) specified by 
        args tomo_ids, class_names and class_numbers,
        respectivly. If a selection argument is None, no selection is
        performed according to that criterion.

        Additional selection can be done using arg rules, a dict that
        contains column names as keys and the values specify the 
        coresponding column elements.

        If arg update is False, modifies the current instance so that only the 
        selected tomos and particles are retained in self.tomos and 
        self.particles. If arg update is True, the current instance is not 
        modified and a new instance is created where attributes tomos and 
        particles contain only the selected tomos and particles. 

        If the class name and number selection leads to the particle table
        not having any particles from one of the tomos, but this tomo is 
        not explicitly selected out by arg tomo_ids, this tomogram is
        retained in the tomos table.

        If arg update is False, returns a new instance of this class where
        attributes tomos and particles are set. However, attributes of 
        this instance (self) that were set after this instance was 
        created are not passes from self to the returned instance.

        Arguments:
          - tomo_ids: (iterable) tomo ids, as given in
          column tomo_id_col of self.particles
          - class_names: (iterable) particle class name, as given in
          column self.class_name_col of self.particles
          - class_numbers: (iterable) particle class number, as given in
          column self.class_number_col of self.particles, or None
          - rules: (dict) selection rules {column_name: values}
          - update: (bool) flag indicating if the current instance is modified,
          or a new instace is returned

        Returns a new instance of this class if update=False, otherwise
        does not return anything
        """

        tomos = self.tomos
        particles = self.particles
        
        # tomo ids
        if tomo_ids is not None:
            tomos = tomos[
                tomos[self.tomo_id_col].isin(tomo_ids)]
            particles = particles[
                particles[self.tomo_id_col].isin(tomo_ids)]

        # combine rules and explicit classe
        rules_default = dict(
            (col, value) for col, value
            in zip([self.class_name_col, self.class_number_col],
                   [class_names, class_numbers])
            if value is not None)
        if rules is not None:
            rules.update(rules_default)
        else:
            rules = rules_default
            
        # select
        for col, value in rules.items():
            particles = particles[particles[col].isin(value)]
            
        # class names
        #if class_names is not None:
        #    particles = particles[
        #        particles[self.class_name_col].isin(class_names)]
            
        # class numbers
        #if class_numbers is not None:
        #    particles = particles[
        #        particles[self.class_number_col].isin(class_numbers)]

        # update this or make another instance
        if update:
            self.tomos = tomos
            self.particles = particles
        else:
            inst = self.__class__()
            inst.tomos = tomos.copy()
            inst.particles = particles.copy()
            return inst

    def select_by_classes(self, set_name, update=False, check=False):
        """Select particles that belong to specified classes.

        Checks all elements of classification columns (listed in 
        self.classification_cols in order to find the specified 
        class(es) (arg set_name). 

        Important: It is advisable that all classes, taken together from 
        all classifications have unique names (can be enforsed by arg 
        check=True). This is because in the case specified class name(s) 
        (arg set_names) exist in multiple classifications (in multiple 
        classification columns), particles from multiple classifications
        are returned.

        Arguments:
          - set_name: (single str or list of strings) name of one or 
          more particle classes (sets)
          - update: (default False) indicates whether this object 
          should be updated
          - check: flag indicating if it should be checked that 
          all classes (sets) from all classification columns 
          (self.classification_cols) have unique names 

        Returns an instance of this class that contains only particles
        of the specified classes (if update=False). Otherwise updates
        this instance.
        """

        rules = self.find_classification(set_name=set_name, check=check) 
        result = self.select(rules=rules, update=update)

        return result

    def find_classification(
            self, set_name, classification_cols=None, check=False):
        """Find classifications that correspond to specified classes.

        Arguments:
          - set_name: name of one or more particle sets (classes)
          - classification_cols: (list) classification columns, or
          self.classification_cols if None (default)
          - check: flag indicating if it should be checked that 
          different classification columns (self.classification_cols)
          do not have common elements (classes, or set names)
          
        Returns: (dict) classification - class relationship
        {classification column: set_name}
        """

        if not isinstance(set_name, (list, tuple, np.ndarray)):
            set_name = [set_name]        
        particles = self.particles

        if classification_cols is None:
            classification_cols = self.classification_cols
        
        # check if classes unique
        if check:
            self.check_classes_unique(
                verbose=False, classification_cols=classification_cols)

        # find columns containing cpecified classes
        rules = {}
        for col in classification_cols:
            existing_classes = particles[col].dropna().unique()
            for set_na in set_name:
                if set_na in existing_classes:
                    try:
                        rules[col].append(set_na)
                    except KeyError:
                        rules[col] = [set_na]

        return rules

    def check_classes_unique(
            self, particles=None, classification_cols=None, verbose=True):
        """Checks whether a class exists in multiple classification columns.

        Raises ValueError if a class is found in multiple classification 
        columns. Otherwise, prints a success message if verbose is True.

        Arguments:
          - particles: (pandas.DataFrame) particles table, or self.particles
          if None (default)
          - classification_cols: (list) classification columns, or
          self.classification_cols if None (default)
          - verbose: if True prints a success message
        """

        if particles is None:
            particles = self.particles
        if classification_cols is None:
            classification_cols = self.classification_cols
            
        # get classes for each classification column
        sets_of_classes = [
            particles[col].dropna().unique()
            for col in classification_cols]

        # check all column pairs
        for cls1, cls2 in itertools.combinations(sets_of_classes, 2):
            if len(np.intersect1d(cls1, cls2)) > 0:
                raise ValueError(
                    "Some classes appear in multiple classification: "
                    + f"classification 1: {cls1}, classification 2: {cls2}")

        else:
            if verbose:
                print("Sucess: No classes in multiple classification columns ")
