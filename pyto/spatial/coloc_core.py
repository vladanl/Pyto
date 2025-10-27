"""
Ccolocalization for one tomo (dataset) and one coloclization case.
 
# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"

from copy import copy, deepcopy

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.cluster.hierarchy import DisjointSet
from scipy.ndimage import distance_transform_edt, find_objects
import pandas as pd 

from . import point_pattern
from .bare_coloc import BareColoc
from . import coloc_functions as col_func


class ColocCore:
    """Class used to generate colocalization for one tomo (dataset).

    Coloclization is calculated using method make().

    """

    def __init__(
            self, full_coloc=True, keep_dist=False, mode='less', pixel_nm=1, 
            metric='euclidean', columns=True, column_factor=2,
            prefix='pattern', suffix='data', n_columns_mode='dist'):
        """Saves arguments.

        If arg n_columns_mode is 'dist' or 'disjoint', columns are defined 
        as subcolumns whose centers are < (or <=) than columns distance 
        apart (columns_distance = columns_factor * colocalization_distance).

        If arg n_columns_mode is 'image', columns are defined as subcolumns
        that touch each other. In this case, arg column_factor is ignored
        (it is effectively set to 2).

        Arguments:
          - full_coloc: if True (default), when 3-colocalization is
          calculated, also the 2-colocalizations between the first pattern
          and each of the other patterns is also calculated  
          - mode: coloclization mode, 'less' or 'less_eq'

          - suffix: suffix added to the coloclization data attribute names

        """

        self.full_coloc = full_coloc
        self.keep_dist = keep_dist
        self.mode = mode
        self.pixel_nm = pixel_nm
        self.metric = metric
        self.columns = columns
        #self.region = region
        self.column_factor = column_factor
        self.prefix = prefix
        self.suffix = suffix
        self.n_columns_mode = n_columns_mode

        #self.data_names = set()
        self.data_names = []

    def make(
            self, patterns, distance, region=None, tomo_id='tomo', names=None,
            suffix=None):
        """Calculate colocalization between point patterns (particle sets).

        First, distances between points of pattern 0 and all other patterns 
        are calculated (using scipy.spatial.distance.cdist()). These
        distances are used to calculate colocalization at multiple 
        colocalization distances (arg distance).

        Elements of args patterns and names have to correspond to each other.
        If arg names is None, ['pattern0', 'pattern1', ...] is used. 

        Each name (element of arg names) has to be a string that is a valid
        Python variable name and it should not contain '_'.

        Arguments:
          - patterns: list of point patterns (length n_patterns), where each
          pattern is an ndarray of shape (n_points, n_dim) containing point 
          pattern coordinates in pixels (thus ints)
          - names: (list of length n_patterns) names of point pattern (particle)
          - distances: list of colocalization distances in nm

        Sets attributes:
          - Attributes pointing to the calculated colocalizations.
          For example, if names=['patternX', 'patternY', 'patternZ'], the
          attributes are:
            - self.patternX_patternY_patternZ+suffix: 3-colocalization
            - self.patternX_patternY+suffix: 2-colocalization
            - self.patternX_patternZ+suffix: 2-colocalization
          - data_names: (list) names of colocalization tables. For the above
          example, this attribute is:
            [patternX_patternY_patternZ+suffix, patternX_patternY+suffix,
            patternX_patternZ] 
          - self.dist_nm: (list of ndarrays) distances between particles in nm.
          the length of the list is n_patterns - 1. Elements of the list 
          contain distances between the first listed pattern (patternX, in 
          the above example) and all other patterns (patternY and patternZ).
          The ndarrays have shape n_points_in_the_first_pattern, 
          n_points_in_the_other_pattern. 
        """

        if suffix is None:
            suffix = self.suffix
        n_patterns = len(patterns)
        if names is None:
            names = [f'{self.prefix}{i}' for i in range(n_patterns)]

        # prepare coloc results arrays for 3-colocalizations
        if (patterns[0] is None) or (patterns[0].size == 0):
            p0_len = 0
        else:
           p0_len = patterns[0].shape[0]
        di_len = len(distance)
        coloc3 = np.zeros(shape=(di_len, p0_len), dtype=bool)
        coloc3_n = np.zeros(shape=(di_len), dtype=int) - 1
        particles3_n = np.zeros(shape=(di_len, n_patterns), dtype=int) - 1
        coloc2 = np.zeros(
            shape=(di_len, n_patterns-1, p0_len), dtype=bool)
        coloc2_n = np.zeros(shape=(di_len, n_patterns-1), dtype=bool) - 1
        particles2_n = np.zeros(shape=(di_len, n_patterns-1, 2), dtype=int) - 1
        n_columns_2 = np.zeros(shape=(di_len, n_patterns-1), dtype=int) - 1
        n_columns_3 = np.zeros(di_len, dtype=int) - 1
        # Note: could not add dtype=int, because in some casses setting to None
        column_size_3 = np.zeros(di_len) - 1
        column_size_2 = np.zeros(shape=(di_len, n_patterns-1)) - 1

        # modify the above for 2-colocalizations
        if n_patterns == 2:
            coloc2 = coloc2.reshape(di_len, p0_len)
            coloc2_n = coloc2_n.reshape(di_len)
            particles2_n = particles2_n.reshape(di_len, 2)
            n_columns_2 = n_columns_2.reshape(di_len)
            column_size_2 = column_size_2.reshape(di_len)
            
        # calculate distances in nm (takes care of None or empty patterns)
        bare = BareColoc(
            mode=self.mode, pixel_nm=self.pixel_nm, metric=self.metric)
        bare.calculate_distances(patterns=patterns)
        if self.keep_dist:
            self.dist_nm = bare.dist_nm_full

        # calculate colocalizations
        self.bare_multid = {}
        for ind, di in enumerate(distance):
            bare.calculate_coloc(distance=di)

            # to save
            self.bare_multid[di] = deepcopy(bare)
            
            # parse coloc results
            if n_patterns > 2:
                coloc3[ind, :] = bare.coloc3
                coloc3_n[ind] = bare.coloc3_n
                particles3_n[ind, :] = bare.particles3_n
                coloc2[ind] = bare.coloc2
                coloc2_n[ind] = bare.coloc2_n
                particles2_n[ind] = bare.particles2_n
            else:
                coloc2[ind] = bare.coloc2[0]
                coloc2_n[ind] = bare.coloc2_n[0]
                particles2_n[ind] = bare.particles2_n[0]

            # find columns
            if self.columns:
                if n_patterns > 2:
                    n_columns_3[ind], column_size_3[ind] = self.find_columns(
                        pattern=patterns[0], coloc=coloc3[ind],
                        distance=di, col_distance=self.column_factor*di,
                        region=region)
                n_columns_2[ind], column_size_2[ind] = self.find_columns(
                    pattern=patterns[0], coloc=coloc2[ind],
                    distance=di, col_distance=self.column_factor*di,
                    region=region)

                #find points in columns

        # find total numbers
        coloc3_total = (
            [self.get_n_points(pat) for pat in patterns]
            * np.ones((di_len, n_patterns), dtype=int))
        
        # make 3-coloc table
        if region is not None:
            size_region = (region > 0).sum()
        if n_patterns > 2:
            df_n0 = pd.DataFrame(
                {'distance': distance, 'id': tomo_id, 'n_subcol': coloc3_n})
            col3_names = [f'n_{pat}_subcol' for pat in names]
            df_n1 = pd.DataFrame(particles3_n, columns=col3_names)
            col3_names_total = [f'n_{pat}_total' for pat in names]
            df_tot = pd.DataFrame(coloc3_total, columns=col3_names_total)
            df = pd.concat((df_n0, df_n1, df_tot), axis=1)
            if region is not None:
                df['size_region'] = size_region
            attr_name = col_func.make_name(names=names, suffix=suffix)
            if self.columns:
                df['n_col'] = n_columns_3
                if region is not None:
                    df['size_col'] = column_size_3
            setattr(self, attr_name, df)
            if attr_name not in self.data_names:
                self.data_names.append(attr_name)
            
        # make 2-coloc table(s)
        coloc2_n0 = coloc2.sum(axis=-1)
        if n_patterns > 2:
            for pat_ind, nam in enumerate(names[1:]):
                df_n0 = pd.DataFrame(
                    {'distance': distance, 'id': tomo_id,
                     'n_subcol': coloc2_n[:, pat_ind],
                     f'n_{names[0]}_subcol': particles2_n[:, pat_ind, 0],
                     f'n_{nam}_subcol': particles2_n[:, pat_ind, 1],
                     f'n_{names[0]}_total': coloc3_total[:, 0],
                     f'n_{names[pat_ind+1]}_total': coloc3_total[:, pat_ind+1]})
                if region is not None:
                    df_n0['size_region'] = size_region
                if self.columns:
                    df_n0['n_col'] = n_columns_2[:, pat_ind]
                    if region is not None:
                        df_n0['size_col'] = column_size_2[:, pat_ind]
                attr_name = col_func.make_name(
                    names=[names[0], nam], suffix=suffix)
                setattr(self, attr_name, df_n0)
                if attr_name not in self.data_names:
                    self.data_names.append(attr_name)

        else:
            df_n0 = pd.DataFrame(
                {'distance': distance, 'id': tomo_id, 'n_subcol': coloc2_n,
                 f'n_{names[0]}_subcol': particles2_n[:, 0],
                 f'n_{names[1]}_subcol': particles2_n[:, 1],
                 f'n_{names[0]}_total': coloc3_total[:, 0],
                 f'n_{names[1]}_total': coloc3_total[:, 1]})
            if region is not None:
                df_n0['size_region'] = size_region
            if self.columns:
                df_n0['n_col'] = n_columns_2
                if region is not None:
                    df_n0['size_col'] = column_size_2
            attr_name = col_func.make_name(names=names, suffix=suffix)
            setattr(self, attr_name, df_n0)
            if attr_name not in self.data_names:
                self.data_names.append(attr_name)

    def calculate_single(self, dist_full, distance):
        """Calculates colocalization data for one colocalization distance.
        
        Uses the precalculated distances between points points of pattern 0
        and all other patterns (arg dist_full). Calculated using
        scipy.spatial.distance.cdist.

        3- and 2-colocalizations are distinguished by the number of elements
        of arg dist_full (2 and 1, respectively). More generally, n-1 elements
        of arg dist_full indicate n-colocalization.

        In case a point set contains no elements, arg dist_full has to have
        shape 0 for the corresponding axis. For example:
          - if pattern1 has no elements:
            dist_full[0] = np.array([]).reshape(len(pattern0), 0)
          - if pattern0 has no elements:
            dist_full[i-1] = np.array([]).reshape(0, len(pattern_i))

        Arguments:
          - dist_full: (list of ndarrays, where the length of the list is 
          n_patterns-1 and the shape of element i of the list is
          len(pattern 0) * len(pattern i) Distances between points.
          - distance: single colocalization distance in nm

        Returns in case of a 3-colocalization (similar for >3-colocalizations):
          - coloc3: (length = len(pattern_0)) Flags showing if the 
          corresponding elements of pattern 0 make 3-colocalizations
          - coloc3_n1: (length n_patterns - 1): Number of elements of 
          patterns 1 and 2 that are in the 3-colocalizations 
          - coloc2: (shape n_patterns-1, len(pattern_0): Flags showing if
          the corresponding elements of pattern 0 make 2-colocalizations
          with patterns 1 and 2, respectively
          - coloc2_n1: (length n_patterns-1) Number of elements of patterns 1 
          and 2 that are in 2-colocalizations with pattern 0 

        Returns in case of 2-colocalization:
        - coloc2: (length = len(pattern_0): Flags showing if
          the corresponding elements of pattern 0 make 2-colocalizations
          with pattern 1
        - coloc2_n1: (single int) Number of elements of pattern 1 
          that are in 2-colocalization with pattern 0 

        """

        # find elements that satisfy distance condition
        if self.mode == 'less':
            dist_conditions = [dist < distance for dist in dist_full]
        elif self.mode == 'less_eq':
            dist_conditions = [dist <= distance for dist in dist_full]
        else:
            raise ValueError(
                f"Argument mode: {self.mode} can be 'less' or 'less_eq'.") 

        # 2-colocalizations
        coloc2 = np.array([
            np.logical_or.reduce(dist_cond, axis=1)
            for dist_cond in dist_conditions])
        coloc2_n1 = np.array([
            np.logical_or.reduce(dist_cond, axis=0).sum()
            for dist_cond in dist_conditions])

        if len(dist_full) == 1:
            return coloc2[0], coloc2_n1[0]

        # 3 (or higher)-colocalizations
        coloc3 = np.logical_and.reduce(coloc2, axis=0)
        
        coloc3_n1 = np.array(
            [np.logical_or.reduce(dist_cond[coloc3], axis=0).sum()
             for dist_cond in dist_conditions]) 

        return coloc3, coloc3_n1, coloc2, coloc2_n1

    def find_columns(self, pattern, coloc, distance, col_distance, region=None):
        """Finds number of columns

        Arguments:
          - pattern: (ndarray n_points x n_dim) point coordinates of the pattern
          that defines columns
          - coloc: (bool, can be 1d ndarray of length=len(pattern), or 2d ndarray
          of shape (n_other_patterns, len(pattern))) True indicates
          points that define subcolumns
          - distance: single colocalization distance [nm]
          - col_distance: column separation distance [nm]

        Return (n_columns, column_size):
          - n_columns: number of columns
          - column_size: Size of all columns together in pixels
        The size of the returned variables depends on the shape of arg coloc. 
        If arg coloc is 1d, the retured values are single numbers, and if 
        it is 2d, the length of returned vars is n_other_pattern
        """

        # deal with 0 points in pattern that defines subcolumns
        no_pattern = False
        if (pattern is None) or (pattern.size == 0):
            no_pattern = True
        coloc = np.asarray(coloc)
        coloc_ndim = coloc.ndim
        if no_pattern:
            if coloc_ndim == 1:
                return 0, 0
            elif coloc_ndim == 2:
                return (
                    np.zeros(shape=coloc.shape[0], dtype=int),
                    np.zeros(shape=coloc.shape[0], dtype=int))
            else:
                raise ValueError(
                    f"Something is not right because pattern has no elements ",
                    + f"and coloc shape is {coloc.shape}")
            
        n_columns = []
        col_size = []
        single_coloc = False
        if len(coloc.shape) == 1:
            coloc = coloc[np.newaxis, :]
            single_coloc = True
        for ind in range(coloc.shape[0]):
        
            # get distances between subcolumns
            # Note this part should be moved to get_n_columns_one()
            subcols = pattern[coloc[ind]]
            dist = self.pixel_nm * pdist(subcols, metric=self.metric)
            if self.mode == 'less':
                dist_cond = (dist < col_distance)
            elif self.mode == 'less_eq':
                dist_cond = (dist <= col_distance)

            # get n columns
            if ((self.n_columns_mode == 'dist')
                or (self.n_columns_mode == 'disjoint')):
                n_col = self.get_n_columns_one(subcols=subcols, dist=dist_cond)
                
            # column size
            cs = self.get_column_size(
                pattern=subcols, distance=distance, region=region)
            if self.n_columns_mode == 'image':
                size, n_col = cs
            else:
                size = cs
                
            n_columns.append(n_col)
            col_size.append(size)

        if single_coloc:
            n_columns = n_columns[0]
            col_size = col_size[0]
        if region is None:
            col_size = None

        return n_columns, col_size

    def get_n_columns_one(self, subcols, dist):
        """Calculates number of columns for one distance.

        """

        # find subcolumn pairs that are close (given by indices of subcols)
        trinds = np.array(np.triu_indices(subcols.shape[0], k=1)).transpose()
        close = np.nonzero(dist)[0]
        pairs = trinds[close, :]

        # find disjoint sets
        if self.n_columns_mode == 'disjoint':
            ds = DisjointSet(range(subcols.shape[0]))
            for co in pairs:
                ds.merge(*co)
            n_columns = len(ds.subsets())

        # based on pairs of distances
        if self.n_columns_mode == 'dist':
            n_subcols = subcols.shape[0]
            col_di = dict(zip(range(n_subcols), range(n_subcols)))
            pairs = np.asarray(pairs)
            pairs.sort()
            for pa in pairs:
                col_di[pa[1]] = col_di[pa[0]] 
            final_columns = np.unique(list(col_di.values()))        
            n_columns = len(final_columns)

        return n_columns
    
    def get_column_size(self, pattern, distance, region=None):
        """Calculates total column size

        Note: The specified image is expected to be already cut as 
        small as possible to still contain the region.  

        Assumes that column_distance = 2 * colocalization_distance.

        ToDo: use colocalization_distance * self.column_factor / 2 instead
        of colocalization distance (arg distance).

        Arguments:
          - pattern: (ndarray n_points x n_dim) coordinates of points that 
          define colocalizations (subcolumns)  
          - region (ndarray): Label image where elements >0 or True designate
          the region and 0 or False background.
          - coloc
          - distance: colocalization distance [nm] (likely different from 
          column distance)
        """

        # sanity checks
        if (region is None) and (self.n_columns_mode == 'image'):
            raise ValueError(
                "When argument region is None, self.n_columns_mode should "
                + "not be 'image'.")
        elif region is None:
            return None
        if pattern.shape[0] == 0:
            if self.n_columns_mode == 'image':
                return 0, 0
            else:
                return 0

        # put subcolumn positions in image
        columns = np.ones_like(region, dtype=float)
        for point in pattern:
            columns[tuple(point)] = 0.

        # get all points closer than distance
        dist = distance_transform_edt(columns)
        distance_pixel = distance / self.pixel_nm
        if self.mode == 'less':
            dist_condition = (dist < distance_pixel)
        elif self.mode == 'less_eq':
            dist_condition = (dist <= distance_pixel)

        # get size
        dist_condition = ((region > 0) & dist_condition)    
        size = dist_condition.sum()

        # find points of all patterns inside (sub) columns
        # or save dist_conditon to be used from make()
        
        # calculate n_columns if image mode
        if self.n_columns_mode == 'image':
            # Not the same as other methods
            _, n_columns = sp.ndimage.label(dist_condition)
            return size, n_columns
        else:
            return size

    @staticmethod
    def get_n_points(pattern):
        """Returns number of points in a pattern

        Argument:
          - pattern: point pattern can be 2d ndarray (shape of any exis 
          can be 0 or higher) or None  
        """
        if (pattern is None) or (pattern.size == 0):
            n_points = 0
        else:
            n_points = pattern.shape[0]
        return n_points
        
