"""
The central part of colocalization.

# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist, pdist, squareform


class BareColoc():
    """Calculate the central part of colocalization and keep results.
    
    Given N point patterns, the N-colocalization contains of all points of 
    the pattern 0 (the main pattern) that are within the colocalization 
    distance of at least one point from each of the patterns 1 - N 
    (other sets). It also contains all points from the other patterns that 
    are within the colocalization distance to the points of the main 
    pattern that belong to the N-colocalization.

    In addition to the N-colocalization, 2-colocalizations between the
    main and all other patterns are determined in te same way as explained
    above. Consequently there are N_patterns-1 2-colocalizations.

    Terms point pattern and particle set are used interchangedly.

    Meant for 2- and 3-colocalizations, but should work for any higher
    colocalization. "2" in the docs and variable names indicates 
    2-colocalization, while "3" stands for 3- or a higher colocalization. 

    Methods:
      - calculate_distances(): given N point patterns, calculates distances 
      between points of the main pattern (pattern 0) and points off all 
      other patterns, resultig in N-pattern-1 distance matrices
      - calculate_coloc(): finds the N-colocalization and 2-colocalizations
      between the main and all other patterns

    Common usage:
      bare = BareColoc()
      bare.calculate_distances(patterns=particle_coords)
      bare.calculate_coloc(distance=distance)

    Attributes set that do not depend on colocalization distance:
      - self.dist_nm_full: (list of length n_patterns, where elements
      are 2d ndarrays) Distances between points, where 
      self.dist_nm_full[k][p, q] is the distance between points p and q
      of pattern k 

    3-colocalization attributes (do not depend on colocalization distance):
      - self.coloc3: (bool 1d ndarray, length len(pattern_0)) Shows
      which points of pattern_0 define 3-colocalizations
      - self.coloc3_indices: like self.coloc3 but shows indices of 
      pattern_0 points that define 3-colocalizations
      - self.coloc3_n: Number of 3-colocalizations
      - self.particles3: (list of length n_patterns where element k is
      bool 1d ndarray of size len(pattern_k)). Shows which points
      belong to 3-colocalizations, self.coloc3[k][i] corresponds to
      point i of pattern_k.
      - self.particles3: like self.particles3 but shows indices of 
      points that belong to 3-colocalizations
      - self.particles3_n: Number of points in 3-colocalizations

    2-colocalization attributes (do not depend on colocalization distance):
      - self.coloc2: (list of length n_patterns-1, where elements are 
      bool 1d ndarrays of length len(pattern_0)) Shows which points of 
      pattern_0 define 2-colocalizations, where self.colc2[k][p] shows 
      if point p of pattern k+1 defines a 2-colocalization between patterns
      0 and k+1.
      - self.coloc2_indices: like self.coloc2 but shows indices of 
      pattern_0 points that define 2-colocalizations
      - self.particles2: (list of length n_patterns-1, where elements 
      are lists of 2 elements and where these elements are ind 1d
      ndarrays) Shows which points belong to 2-colocalizations. Elements
      of the outermost list correspond to 2-colocalizations between 
      pattern_0 and other patterns. Specifically:
        - self.colc2[k][0][p] shows if point p of pattern_0 belongs to 
        the 2-colocalization between patterns 0 and k+1
        - self.colc2[k][1][p] shows if point p of pattern_(k+1) belongs to 
        the 2-colocalization between patterns 0 and k+1
      - self.particles2_indices: like self.particles2 but shows indices of 
      points that belong to 2-colocalizations
      - self.particles2_n: like self.particles2 but shows number of 
      points in 2-colocalizations
        

    """

    def __init__(self, mode='less', pixel_nm=1, metric='euclidean', ):
        """Sets attributes from arguments
        """

        self.mode = mode
        self.pixel_nm = pixel_nm
        self.metric = metric

    @property
    def coloc3_indices(self):
        """Indices of 3-colocalizations.

        1d int ndarray. Directly corresponds to self.coloc3
        """
        return self.coloc3.nonzero()[0]

    @property
    def particles3_indices(self):
        """Indices of points in the 3-colocalization.

        List (length n_patterns) of 1d int ndarrays. Directly corresponds 
        to self.particles3
        """
        return [x.nonzero()[0] for x in self.particles3]

    @property
    def coloc3_n(self):
        """Number of 3-colocalizations.

        """
        return self.coloc3.sum()
        
    @property
    def particles3_n(self):
        """Number of points in 3-colocalizations.

        """
        return [x.sum() for x in self.particles3]
        
    @property
    def coloc2_indices(self):
        """Indices of all 2-colocalization.

        List of length n_patterns-1, where elements are int 1d 
        ndarrays. Directly corresponds to self.coloc2, except that the
        ndarrays contain indices of points that define
        colocalization.
        """
        return [x.nonzero()[0] for x in self.coloc2]

    @property
    def particles2_indices(self):
        """Indices of points in all 2-colocalization.

        Nested list of shape (n_patterns-1, 2), where elements are int 1d 
        ndarrays. Directly corresponds to self.coloc2, except that the
        (innermost) ndarrays contain indices of points that are in a
        colocalization.
        """
        return [[x.nonzero()[0] for x in other] for other in self.particles2]

    @property
    def coloc2_n(self):
        """Number of colocalizations in all 2-colocalizations.

        """
        return [x.sum() for x in self.coloc2]
         
    @property
    def particles2_n(self):
        """Number of colocalizations in all 2-colocalizations.

        """
        return [[x.sum() for x in other] for other in self.particles2]
         
    def calculate_distances(self, patterns):
        """Calculates distances between point patterns.

        Calculates distances between points of pattern 0 (arg patterns[0])
        and each of the other patterns (patterns[1], ...) in nm. Because
        point coordinates (arg patterns) are specified in pixels, uses
        self.pixel_nm to convert distances to nm. Uses   
        sp.spatial.distance.cdist for actual distance calculations.

        Sets self.dist_nm_full: 
          (list of length n_patterns - 1) where element
          k contains distances between pattern 0 and pattern k+1 in nm 
          as an ndarray of shape (len(patterns[0]), len(patterns[k+1])

        If one of the patterns is None or np.array([]), self.dist_nm_full
        will still be a n_patterns-1 long list of nparrays, but the
        length of these arrays corresponding to the non-existing pattern
        will be 0.

        Even if all patterns contain no points, the above will hold.

        Argument:
          - patterns: (list of length 2+) coordinates of multiple point 
          patterns (particle sets), where each elelment of this argument 
          defines one pattern and is specified as ndarray of shape 
          (n_poins, n_dim) Empty sets can be specified as None or 
          nm.array([]).
        """

        n_patterns = len(patterns)
        
        # get n points of each pattern
        patterns_len = []
        for pat_ind in range(n_patterns):
            if (patterns[pat_ind] is None) or (patterns[pat_ind].size == 0):
                patterns_len.append(0)
            else:
                patterns_len.append(patterns[pat_ind].shape[0])

        # deal with pattern[0] containing no points  
        if patterns_len[0] == 0:
            self.dist_nm_full = [
                np.array([]).reshape(0, patterns_len[pat_ind])
                for pat_ind in range(n_patterns)]
            return

        # calculate distances when patterns[0] has >0 points
        dist_nm_full = []
        for pat_ind in range(n_patterns):
            if patterns_len[pat_ind] == 0:
                dist_nm_full.append(np.array([]).reshape(
                    patterns[0].shape[0], 0))
            else:
                dist_nm_full.append(
                    self.pixel_nm * sp.spatial.distance.cdist(
                        patterns[0], patterns[pat_ind], metric=self.metric))
        self.dist_nm_full = dist_nm_full

        return

    def calculate_coloc(self, distance):
        """Calculates colocalization data for one colocalization distance.
        
        Uses the precalculated distances between points points of pattern 0
        and all other patterns (arg dist_full). Calculated using
        scipy.spatial.distance.cdist.

        3- and 2-colocalizations are distinguished by the number of elements
        of self.dist_nm_full (2 and 1, respectively). More generally, n-1 
        elements of self.dist_nm_full indicate n-colocalization.

        In case a point set contains no elements, self.dist_nm_full has to have
        shape 0 for the corresponding axis. For example:
          - if pattern1 has no elements:
            dist_full[0] = np.array([]).reshape(len(pattern0), 0)
          - if pattern0 has no elements:
            dist_full[i-1] = np.array([]).reshape(0, len(pattern_i))

        Arguments:
          - distance: single colocalization distance in nm

        Sets attributes:
          to continue

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
          the corresponding elements of pattern 0 makes 2-colocalizations
          with pattern 1
        - coloc2_n1: (single int) Number of elements of pattern 1 
          that are in 2-colocalization with pattern 0 

        """

        # find elements that satisfy distance condition
        if self.mode == 'less':
            dist_conditions = [dist < distance for dist in self.dist_nm_full]
        elif self.mode == 'less_eq':
            dist_conditions = [dist <= distance for dist in self.dist_nm_full]
        else:
            raise ValueError(
                f"Argument mode: {self.mode} can be 'less' or 'less_eq'.") 

        # 2-colocalizations
        self.coloc2 = [
            np.logical_or.reduce(dist_cond, axis=1)
            for dist_cond in dist_conditions[1:]]

        # paricles in 2-colocalizations
        particles2_other = [
            np.logical_or.reduce(dist_cond, axis=0)
            for dist_cond in dist_conditions[1:]]
        coloc2_ax = [c2.reshape(-1, 1) for c2 in self.coloc2]
        particles2_main = [
            np.logical_or.reduce(c2_ax & dist_cond, axis=1)
            for c2_ax, dist_cond in zip(coloc2_ax, dist_conditions[1:])] 
        self.particles2 = [
            [main, other]
            for main, other in zip(particles2_main, particles2_other)]
        
        if len(self.dist_nm_full) == 2:
            return 

        # main points in 3 (or higher)-colocalizations
        self.coloc3 = np.logical_and.reduce(np.asarray(self.coloc2), axis=0)
        coloc3_ax = self.coloc3.reshape(-1, 1)

        # other points in 3 (or higher)-colocalizations
        self.particles3 = [
            np.logical_or.reduce(coloc3_ax & dist_cond, axis=0)
            for dist_cond in dist_conditions]

        


