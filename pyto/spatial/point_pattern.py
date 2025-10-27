"""
Functions to make random point patterns.

Reandom point patterns are specified as int type coordinates on specified
regions. Regions can be rectangular or arbitrary, defined as an array
(image) or a list of region coordinates.

The main functionality of this module are the following:

1) Generate random patterns with or without imposing a minimal (exclusion)
distance between points

  random_region(N=N_points, region=region_image, region_id=region_label_id,
                exclusion=exclusion_distance)

  Instead of specifying an image, a region can be specified by
  an array of (int) coordinates that comprise the region:

  random_region(N=N_points, region_coords=region_coordinates,
                exclusion=exclusion_distance)

  To make another point pattern with exclusion:
    - define a function that generates the desired point pattern without 
    exclusion
    - define a function that passes the above function to pattern_exclusion(),
    which performs exclusion on the pattern produced by the above function
  Function pattern_exclusion() calls a no exclusion pattern generating
  function as many times as needed to generate the specified number
  of points while imposing exclusion.  

2) Generate patterns that partially cluster around (interact with)
specified cluster centers

  cocluster_region(clusters=cluster_centers, N=N_points, p_cluster, max_dist)

3) Project a pattern on a region specified as an image or by coordinates:

  project(points=point_pattern, region)

# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"

from functools import partial
import itertools

import numpy as np
from numpy.random import default_rng
import scipy as sp
from scipy.spatial.distance import cdist, pdist, squareform
try:
    from sympy import Interval, Union, Intersection
except ModuleNotFoundError:
    print(
        "Info: Module sympy could not be loaded. However, this is not "
        + "a problem for running Pyto, because sympy is currently used "
        + "only for development.") 
    
import pyto


#
# Random patterns
#

def random_rectangle(
        N, rectangle, exclusion=None, metric='euclidean', other=None,
        mode='fine', seed=None, rng=None, max_iter=100, n_factor=2):
    """Makes random pattern in n_dim rectangle with exclusion. 
    
    If arg N is a single number, uses random_rectangle_fun() to generate 
    random points on a specified rectangle and pattern_exclusion() to 
    exclude particles. In this case arg rectangle is 2d ndarray that 
    defines one rectangle. 
    
    If arg N is a list or tuple, generates random points on multiple 
    rectangles. In this case arg rectangle hs to be an iterable of 2d 
    ndarrays. Elements of N and rectangles have to correspond to each 
    other.

    If arg other is given, the generated points cannot be closer than
    arg exclusion to the points specified by arg other.

    Arg mode determines the algorithm used to exclude points. Both implemented
    methods generate the specified number of particles having a correct
    exclusion. For details, see pattern_exclusion() doc.

    If it is not possible to generate points as specified (by args N, 
    exclusion, rectangle, max_iter), ValueError is raised.

    Arguments:
      - N: (single number for one, or list or tuple of numbers for
      multiple rectangles) number of points to be generated in the /
      each of rectangles
      - rectangle: (ndarray 2 x n_dim for one rectangle, or an iterable
      of 2 x n_dim arrays for multiple rectangles) rectangle(s) where the 
      low coordinates are specified by (one_)rectangle[0, :] and the high 
      coords by (one_)rectangle[1, :]
      - exclusion: exclusion distance in pixels
      - other: (ndarray, n_other_points x n_dim) coordinates of additional 
      points, the generated points cannot be closer than (arg) exclusion to 
      the additiona points
      - metric: distance calculation metric
      - mode: method to calculate exclusion, 'rough' or 'fine'
      - rng: random number generator, if None a new one is created
      - seed: random number generator seed used in case a new random
      number generator is created here
      - max_iter: max number of iterations, where each iteration comprises
      generating eand excluding points
      - n_factor: (int >= 1) in each iteration n_factor * N new points
      are generated before exclusion

    Returns: random points (int ndarray, N x n_dim), with exclusion 
    """

    if rng is None:
        rng = default_rng(seed=seed)

    if not isinstance(N, (list, tuple)):
        N = [N]
        rectangle = [rectangle]

    values = []
    for n_one, rctg in zip(N, rectangle):
        
        fun_partial = partial(random_rectangle_fun, rectangle=rctg)
        vals = pattern_exclusion(
            pattern_fun=fun_partial, N=n_one, exclusion=exclusion,
            metric=metric, other=other, mode=mode, seed=seed, rng=rng,
            max_iter=max_iter, n_factor=n_factor)
        values.extend(vals)

    result = np.vstack(values)
    return result

def random_rectangle_fun(rng, N, rectangle):
    """Generates random points in a n_dim rectangle without exclusion.

    Meant to be used in random_rectangle(), where arg rectangle is fixed 
    and it is then passed to pattern_exclusion(). Can be also used as
    a standalone function.

    Arguments:
      - rng: random number generator:
      - N: number of points to be generated
      - rectangle: (ndarray, 2 x n_dim) rectangle where the low coordinates
      are specified by rectangle[0, :] and the high coords by rectangle[1, :]

    Returns: random integers (ndarray, N x n_dim)
    """

    rectangle = np.asarray(rectangle)
    ndim = rectangle.shape[1]
    vals = rng.integers(
        low=rectangle[0, :], high=rectangle[1, :], size=(N, ndim))
    
    return vals
    
def random_region(
        N, region=None, region_id=None, region_coords=None, exclusion=None,
        metric='euclidean', other=None, mode='fine', shuffle=True,
        seed=None, rng=None, max_iter=100, n_factor=2):
    """Makes an random point pattern on an arbitrary region with exclusion.

    The region can be specified by a label image (args region and region_id),
    or by coordinates of all points of the region (arg region_coords). In the
    former case, labeled points are converted to coordinates. Random points 
    are determined by generating 1d random point set on the index of the
    coordinates. 

    I arg other is given, the generated points cannot be closer than
    arg exclusion to the points specified by arg other.

    Arg mode determines the algorithm used to exclude points. Both implemented
    methods generate the specified number of particles having  a correct
    exclusion. For details, see pattern_exclusion() doc.

    Regarding reproducibility:
      - If shuffle is False and both args rng and seed are None, the
      result changes
      - If seed or rng is given, the result is determined by seed,
      rng and shuffle 
    
    Arguments:
      - N: number of points to be generated
      - region: (ndarray or pyto.core.Image) label image
      - region_id: id of the region in the label image (region)
      - region_coords: (ndarray, n_region_coords x n_dim) coordinates of all 
      points of a region, used if args region and region_id are not specified
      - exclusion: exclusion distance in pixels
      - metric: distance calculation metric
      - other: (ndarray, n_other_points x n_dim) coordinates of additional 
      points, the generated points cannot be closer than (arg) exclusion to 
      the additiona points
      - mode: method to calculate exclusion, 'rough' or 'fine'
      - shuffle: used only to determine if region coordinates from (arg)
      region are shuffled (default True) 
      - rng: random number generator, if None a new one is created
      - seed: random number generator seed used in case a new random
      number generator is created here
      - max_iter: max number of iterations, where each iteration comprises
      generating eand excluding points
      - n_factor: (int >= 1) in each iteration n_factor * N new points
      are generated before exclusion

    Returns (int ndarray N_points x n-dims) Random int coordinate points on 
    the region
    """

    # figure out how the region is specified
    #if (region is not None) and (region_id is not None):

    #    if isinstance(region, pyto.core.Image):
    #        region = region.data
    #    region_coords = np.stack(np.nonzero(region==region_id), axis=1)

    #elif region_coords is not None:
    #    region_coords = np.asarray(region_coords)

    #else:
    #    raise ValueError(
    #        "A region has to be specified by eather an image (args region "
    #        + "and region_id args), or coordinates of all region points "
    #        + "(arg region_coords).")

    # shuffle region coords
    #if rng is None:
    #    rng = default_rng(seed=seed)
    #if shuffle:
    #    rng.shuffle(region_coords)

    region_coords = get_region_coords(
        region=region, region_id=region_id, region_coords=region_coords,
        shuffle=shuffle, seed=seed, rng=rng)
   
    # use random by region coordinates
    #n_coords = region_coords.shape[0]
    fun_partial = partial(random_region_fun, coords=region_coords)
    vals = pattern_exclusion(
        pattern_fun=fun_partial, N=N, exclusion=exclusion, metric=metric,
        other=other, mode=mode, seed=seed, rng=rng, max_iter=max_iter,
        n_factor=n_factor)
    
    return vals

def random_region_fun(rng, N, coords):
    """Generates random points in an arbitrary region without exclusion.

    The region can be of any dimension and shape. It is specified by
    all coordinates that it contains.

    Meant to be used in random_region(), where arg coords is fixed 
    and it is then passed to pattern_exclusion(). Can be also used as
    a standalone function.

    Arguments:
      - rng: random number generator
      - N: number of points to generate
      - coords: (array, shape N_points x n_dims) All coordinates of 
      the region

    Returns (ndarray N_points x n_dims) Random points on the region
    """
    
    n_coords = coords.shape[0]
    region_inds = rng.integers(low=0, high=n_coords, size=N)
    vals = coords[region_inds]    
    return vals
    
def pattern_exclusion(
        pattern_fun, N, exclusion=None, metric='euclidean', other=None,
        mode='fine', seed=None, rng=None, max_iter=100, n_factor=2):
    """Generates point pattern considering exclusion distance on any geometry.

    Takes a specified point pattern generator function as an argument 
    (pattern_fun) to generate a specified number of points (arg N)
    while imposing a minimal exclusion distance (arg exclusion) between
    the generated points. 

    If arg other is given, the generated points cannot be closer than
    arg exclusion to the points specified by arg other.

    If it is not possible to generate points as specified (by N, exclusion,
    and pattern_fun), ValueError is raised.

    In more detail, it iteratively generates n_factor * N points 
    and removes points closer to the minimal exclusion distance as long as it 
    is needed to reach N points. At each iteration, it removes newly generated
    points that are closer than the exclusion distance to the previously
    accepted points, and removes points of the newly generated set that 
    are closer to each other than the exclusion distance.

    Two methods are implemented to exclude poins within a newly generated set 
    (the second of the tasks mentioned above):
      - rough (mode='rough'): When two points are closer than the exclusion 
      distance, it keeps the first and removes the second point. This is 
      simple and fast but the efficiency is not optimal. Namely, when the
      distances beteween points 1 and 2, and between points 2 and 3,
      but not between points 1 and 3 are shorter than the exclusion, points
      2 and 3 will be excluded although 3 should be retained. 
      - fine (mode='fine'): After doing the rough exclusion it recursively 
      cleans the excluded particles to find those that could be retained.
    Therefore, the rough method uses a simple exclusion proocedure that may 
    unnecessarily exclude some points and thus generates more random points,
    while the fine method does precise exclsion (which takes longer time)
    but overall generates a lower number of particles. Importantly, both 
    methods generate random points that obey the exclusion.

    Regarding reproducibility:
      - If both args rng and seed are None, the result changes
      - If rng is not None, the result is determined by the state of rng
      - If rng is None and seed is not None, the result is predictable,
      determined by seed
    
    Arguments:
      - pattern_fun: function that generates a point pattern 
      (takes n_factor*N points and random number generator as arguments)
      - N: number of points to be generated
      - exclusion: exclusion distance in pixels
      - other: (ndarray, n_other_points x n_dim) coordinates of additional 
      points, the generated points cannot be closer than (arg) exclusion to 
      the additiona points
      - metric: distance calculation metric
      - mode: method to calculate exclusion, 'rough' or 'fine'
      - rng: random number generator, if None (default) a new one is created 
      - seed: random number generator seed used in case a new random
      number generator is created here (default None)
      - max_iter: max number of iterations, where each iteration comprises
      generating eand excluding points
      - n_factor: (int >= 1) in each iteration n_factor * N new points
      are generated before exclusion

    Returnes cleaned points (ndarray, n_cleaned_points x n_dim)
    """

    if N == 0:
        return np.array([])
    
    if rng is None:
        rng = default_rng(seed=seed)
    
    # deal with exclusion
    if exclusion is not None:
        n_curr = 0
        N_over = n_factor * N        
        clean = None
        clean_other = None
        if other is not None:
            clean_other = other
        
        # generate random as long as it is needed to reach the specified N
        for _ in range(max_iter):
            
            # make random, exclude with respect to clean, exclude among
            values = pattern_fun(N=N_over, rng=rng)
            if clean_other is not None:
                clean_loc = exclude(
                    points=values, exclusion=exclusion, other=clean_other,
                    mode=mode, metric=metric)
            else:
                clean_loc = values
            clean_loc = exclude(
                points=clean_loc, exclusion=exclusion, mode=mode, metric=metric)
            if len(clean_loc) == 0:
                continue

            # add to previous and count
            if clean is not None:
                clean = np.concatenate((clean, clean_loc), axis=0)
            else:
                clean = clean_loc
            if other is not None:
                clean_other = np.concatenate((clean, other), axis=0)
            else:
                clean_other = clean
            n_curr = clean.shape[0]
            if n_curr >= N:
                break
                
        else:
            raise ValueError(
                f"{N} random points could not be generated in {max_iter} "
                + "iterations. Perhaps the number of iterations should be "
                + "increased, or it is not possible to generate more "
                + "random points considering the specified area and "
                + "exclusion distance.")  
                
        # remove extra
        clean = clean[0:N, :]
        return clean                        
        
    # generate random without exclusion
    else:
        vals = pattern_fun(rng=rng, N=N)
        return vals
     
def exclude(points, exclusion, other=None, metric='euclidean', mode='fine'):
    """Exclude points by Euclidean distance.

    If arg other is None, excludes elements of arg points that are closer to 
    each other than the specified distance (arg exclusion). When two points
    closer than arg exclusion are found, the second one is excluded (according 
    to the order in arg points).

    If arg other is not None, excludes (elements of arg) points that are closer
    to points of another pattern (arg other) than the specified distance (arg 
    exclusion).

    Arg mode determines the algorithm used to exclude points. Both implemented
    methods ensure the correct exclusion. However, while 'rough' method is 
    faster, it may remove particles that could be kept. For details, see 
    pattern_exclusion() doc.

    Arguments:
      - points: (ndarray n_points x n_dim) Coordinates of points in pixels
      (thus int)
      - exclusion: exclusion distance in pixels
      - other: (ndarray n_points x n_dim) Coordinates of another point 
      pattern (in pixels, thus int)
      - metric: distance calculation metric
      - mode: method to calculate exclusion, 'rough' or 'fine'
    
    Returnes cleaned points (ndarray, n_cleanedPoints x n_dim). If there are
    no points (arg points is None or []), np.array([]) is returned.
    """

    if (points is None) or (len(points) == 0):
        return np.array([])
    points = np.asarray(points)
    
    if other is None:
        
        # exclude points close to each other in the rough way
        dists_cond = (pdist(points, metric=metric) < exclusion)
        dists_square = squareform(dists_cond)
        dists_square_tri = np.triu(dists_square, k=0)
        no_go = np.logical_or.reduce(dists_square_tri, axis=0)
        clean = points[np.logical_not(no_go)]

        # do not exclude points that are excluded based on already excluded
        if mode == 'fine':
            dists_square_tri[no_go.nonzero()] = False 
            no_go_fine = np.logical_or.reduce(dists_square_tri, axis=0)
            clean_fine = points[np.logical_not(no_go_fine)]

            possible_go = no_go & np.logical_not(no_go_fine)
            if possible_go.any():
                possible = points[possible_go]
                clean_plus = exclude(
                    points=possible, exclusion=exclusion, other=None,
                    mode='fine', metric=metric)
                clean = np.concatenate((clean, clean_plus), axis=0)

        elif mode != 'rough':
            raise ValueError(f"Argument mode: {mode} can be 'rough' or 'fine'") 
            
    else:
        
        # exclude points close to other
        other = np.asarray(other)
        dists = (cdist(other, points, metric=metric) < exclusion)
        no_go_other = np.logical_or.reduce(dists, axis=0)
        clean = points[np.logical_not(no_go_other)]
    
    return clean

#
# Interacting patterns
#

def cocluster_region(
        clusters, N, p_cluster=1, mode=None, region=None, region_id=None,
        region_coords=None, max_dist=None, exclusion=None,
        metric='euclidean', shuffle=True, seed=None, rng=None):
    """Generates point patterns that cluster around specified cluster centers.

    If mode is '1o1', (arg) N points are randomly assigned to (arg)
    clusters, one particle per cluster at most. In this case N cannot
    be larger than the number of clusters and (arg) p_cluster is
    ignored.

    If mode is None or 'many_to1', each of the N points is randomly
    assigned to one of the clusters with probability (arg) p_cluster.
    In this way, more than one point can be assigned to a cluster and
    the total number of the clustered point is stochastic with
    expectation value N_total_points x p_cluster.

    Points assigned to clusters are randomly placed within the neighborhood 
    of the corresponding cluster centers (arg clusters). The neighborhood
    is defined as a part of the region (specified by args region
    and regin_id) at most (arg) max_dist from the cluster center
    (see random_hoods()).

    Points not assigned to clusters are randomly distributed on the 
    entire (arg) region.

    Points in the resulting pattern are distributed so that the minimal
    distance between them is (arg) exclusion.

    Arguments:
      - clusters: (ndarray, n_clusters x n_dim) cluster center coordinates
      - N: total number of points
      - mode: defines how are points assigned to clusters, 'many_to1' (same
      as None, default) or 'max1'
      - p_cluster: probability that a point belongs to any cluster 
      - region: (ndarray or pyto.core.Image) label image
      - region_id: id of the region in the label image (region)
      - region_coords: (ndarray, n_region_coords x n_dim) coordinates of all 
      points of a region, used if args region and region_id are not specified
      - max_dist: maximal distance of a point to the center of the cluster
      (pixels)
      - exclusion: exclusion distance in pixels
      - metric: distance calculation metric
      - shuffle: flag indicating whether locaition of points assigned
      to clusters and those that are not assigned are randomly
      distributed within the neighborhood and the entire region,
      respectfully (default=True, strongly recommended)
      - rng: random number generator, if None a new one is created, used
      only if shuffle=True (default None)
      - seed: random number generator seed used in case a new random
      number generator is created here, used only if shuffle=True
      (default None)
    
    Return: (ndarray n_points x n_dim) coordinates of points, where the
    points at the beginning are clustered (expected number is N * p_cluster,
    but the actual number is stochastic) and the remaining points are
    not clustered.
    """

    # figure out the number of points to be generated for each cluster
    clusters = np.asarray(clusters)
    n_clusters, n_dim = clusters.shape
    if (mode is None) or (mode == 'many_to1'):
        n_points_in_cluster = get_n_points_cluster(
            n_clusters=n_clusters, n_points=N, p_cluster=p_cluster, seed=seed,
            rng=rng)
    elif (mode == 'max1'):
        n_points_in_cluster = np.zeros(n_clusters, dtype=int)
        n_points = np.round(N * p_cluster).astype(int).tolist()
        if n_points > n_clusters:
            raise ValueError(
                f"In mode 'max1',  the number of points to cluster {n_points} "
                + f"cannot be larger than the number of clusters {n_clusters}") 
        n_points_in_cluster[:n_points] = 1
        if rng is None:
            rng = default_rng(seed=seed)
            rng.shuffle(n_points_in_cluster)
        
    n_clustered_points = n_points_in_cluster.sum()
    n_random_points = N - n_clustered_points
    
    # get region coords
    region_coords = get_region_coords(
        region=region, region_id=region_id, region_coords=region_coords, 
        shuffle=shuffle, seed=seed, rng=rng)

    # distance based neighborhoods
    if max_dist is not None:
        # shuffle=True to make deterministic results (although shuffle
        # used above)
        hood_points = random_hoods(
            clusters=clusters, n_points=n_points_in_cluster,
            region_coords=region_coords, max_dist=max_dist,
            exclusion=exclusion, metric=metric, shuffle=False,
            seed=seed, rng=rng)
        other_points = random_region(
            N=n_random_points, region_coords=region_coords, exclusion=exclusion,
            other=hood_points, metric=metric, shuffle=False,
            seed=seed, rng=rng)
        try:
            points = np.concatenate([hood_points, other_points], axis=0)
        except ValueError:
            points = np.concatenate(
                [hood_points.reshape(-1, n_dim),
                 other_points.reshape(-1, n_dim)],
                axis=0)
        points = points.round().astype(int)
           
    else:
        raise ValueError(
            "Currently only coclustering mehod implemented is based on"
            + " distance neighborhoods, so arg max_dist is required.")
        
    return points
    
def random_hoods(
        clusters, n_points, region_coords, max_dist, exclusion=None,
        metric='euclidean', mode='fine', shuffle=True,
        seed=None, rng=None, max_iter=100, n_factor=2):
    """Makes random int point pattern in multiple circular neighborhoods.

    Randomly assigns the specified number of points (arg n_points) 
    to neighborhoods of cluster centers. Cluster centers are
    specified by arg clusters and the neighborhoods are defined as all
    points of the specified region (arg region_coords) that are located
    up to the specified distance (arg max_dist) to the cluster centers.

    Points in the resulting pattern are distributed so that the minimal
    distance between then is (arg) exclusion.

    If arg shuffle is True, points in neighborhoods are shuffled so
    that the assigned points are random (recommended).
    
    Arguments:
      - clusters: (ndarray, n_clusters x n_dim) cluster center coordinates
      - n_points: (1d array of length n_clusters) number of points in 
      clusters
      - region_coords: (ndarray, n_region_coords x n_dim) coordinates of all 
      points of a region (pixels)
      - max_dist: maximal distance of a point to the center of the cluster
      (pixels)
      - exclusion: exclusion distance in pixels
      - metric: distance calculation metric
      - mode: method to calculate exclusion, 'rough' or 'fine'
      - shuffle: flag indicating whether neighborhood points are shuffled
      (default True, recommended)
      - rng: random number generator, if None a new one is created
      - seed: random number generator seed used in case a new random
      number generator is created here
      - max_iter: max number of iterations, where each iteration comprises
      generating eand excluding points
      - n_factor: (int >= 1) in each iteration n_factor * N new points
      are generated before exclusion
    
    Return: (ndarray n_points x n_dim) coordinates of points  
    """
    
    #  select coords in hoods
    hoods_coords = (cdist(clusters, region_coords, metric=metric) <= max_dist)

    # loop over clusters
    n_clusters, n_dim = np.asarray(clusters).shape
    hood_points_list = []
    for cluster_ind in range(n_clusters):
        reg_indices = np.nonzero(hoods_coords[cluster_ind, :])[0]
        reg_coord = region_coords[reg_indices, :]
        if reg_coord.shape[0] == 0:
            raise ValueError(
                f"Cluster located at {clusters[cluster_ind]} does not have"
                + " a neighborhood on the specified region.")
        
        # make random points in this hood
        hood_points_local = random_region(
            N=n_points[cluster_ind], region_coords=reg_coord,
            exclusion=exclusion, metric=metric, mode=mode, shuffle=shuffle,
            seed=seed, rng=rng, max_iter=max_iter, n_factor=n_factor)
        if (hood_points_local is not None) and (hood_points_local.size > 0):
            hood_points_list.append(hood_points_local)

    if len(hood_points_list) > 0:
        hood_points = np.concatenate(hood_points_list, axis=0)
    else:
        hood_points = np.array([], dtype=int).reshape(0, n_dim)
        
    return hood_points
    
def get_n_points_cluster(n_clusters, n_points, p_cluster, seed=None, rng=None):
    """Assign random number of points to clusters.

    Assigns a random number of points to each of the (arg) n_clusters
    under the constraints:
      - total number of points is (arg) n_points
      - each point has a probability of (arg) p_cluster to belong to
    one of the clusters
      - clusters have the same probability

    Consequently the total number of points assigned to clusters will
    be smaller or equal to the arg n_points.

    The assignment proceeds as:
      - Each point is randomly assigned to one of the clusters
      - Each point is randomly selected (probability p_cluster) to belong
    to clusters on not.

    Each of the (arg) n_points point is randomly assigned, based on 
    the probability (arg) p_cluster), to belong to one of the clusters, or
    not to belong to any of the clusters. Points that belong to clusters
    are randomly assigned (with equal weights) to one of the (arg) 
    n_clusters clusters.

    Arguments:
      - n_clusters: number of clusters
      - n_points: number of points
      - p_cluster: probability that a point belongs to any cluster 
      - rng: random number generator, if None a new one is created
      - seed: random number generator seed used in case a new random

    Returns (numpy.ndarray, len n_clusters) number of points for each 
    cluster
    """

    if rng is None:
        rng = default_rng(seed=seed)

    clusters_init = rng.integers(low=0, high=n_clusters, size=n_points)
    pick_clusters = (rng.random(n_points) <= p_cluster)
    clusters = clusters_init[pick_clusters]
    n_points_cluster, _ = np.histogram(
        clusters, bins=n_clusters, range=(0, n_clusters))

    return n_points_cluster

def colocalize_pattern(
        fixed_pattern, n_colocalize, mode=None,
        fixed_fraction=1, colocalize_fraction=1,
        region=None, region_id=1, region_coords=None, max_dist=0,
        shuffle_fixed=True, shuffle_region=True, rng=None, seed=None):
    """Makes a point pattern that colocalizes (interact with) a given pattern.

    Generates a colocalized point pattern so that a fraction of the
    colocalized points colocalizes with a fraction of the fixed points
    and where all colocalized points are located on the specified region.
    Procedure:
      - Selects the specified fraction of the fixed pattern points
      - Further select points located in the specified region
      - Places the specified fraction of colocalization points (arg
      colocalize_fraction) within neighborhoods of the selected fixed
      pattern points, depending on the arg mode (see below)
      - Places the remaining colocalization points randomly on the
      specified region

    If mode is 'max1', at most one point is colocalized with each
    selected fixed pattern point.

    If mode is 'kd', colocalized fraction is calculated from the
    equilibrium condition as:
      fixed_fraction * fixed_pattern.shape[0] / n_colocalize
    and arg fixed_fraction is ignored. The colocalized fraction should
    not be >1.

    If mode is None, or 'many_to1', multiple points can be colocalized
    with any of the selected fixed points. The actual number of
    colocalized points is stochastic with expectation
    n_colocalize * colocalize_fraction.
    
    If arg shuffle_fixed is True, the selection of fixed points is random.
    If it is False, the first fixed_fraction * n_fixed_pattern_points
    are selected. This is useful when the same subset of fixed points has
    to colocalize with multiple colocalization point sets. In this case,
    one needs to ensure that the specified fixed_pattern ponts are randomly
    distributed by a previous reshuffling.
    
    Similar to cocluster_region(), but differs from it as follows:
      - Colocalized points have to be located within the region
      - Only a fraction of fixed pattern is used for colocalization
    
    Arguments:
      - fixed_pattern: (ndarray n_points x n_dim) fixed point pattern
      - n_colocalize: number of colocalized points to generate
      - mode: defines how are points colocalized with the fixed pattern,
      'many_to1' (same as None, default), 'max1' or 'kd'
      - fixed_fraction: fraction of the fixed points used for
      colocalization (default 1)
      - colocalized_fraction: fraction of the colocalized points that are
      colocalizized (default 1)
      - region: (ndarray or pyto.core.Image) label image (default None)
      - region_id: id of the region in the label image (region) (default 1)
      - region_coords: (ndarray, n_region_coords x n_dim) coordinates of all 
      points of a region, used if args region and region_id are not specified
      (default None)
      - max_dist: distance (in pixels) that defines colocalization
      (interaction) neighborhoods (default 0)
      - shuffle_fixed: flag indicating if fixed_pattern points are
      randomly shuffled before a fraction of them is selected (default
      True)
      - shuffle_region: flag indicating if region coordinates are 
      randomly shuffled, to ensure random placement within neighborhoods
      and on the region (default True, strongly recommended)
      - rng: random number generator for shuffling, if None a new one
      is created
      - seed: random number generator seed for shuffling, used in case
      a new random number generator is created here

    Return: (ndarray n_colocalize x n_dim) colocalized points, where the
    points at the beginning are colocalized (expected number is
    n_colocalize * colocalized_fraction, but the actual number is
    stochastic) and the remaining points are not colocalized.
    """

    # select points from fixed pattern
    fixed = select_by_region(
        pattern=fixed_pattern, region=region, region_id=region_id,
        region_coords=region_coords, fraction=fixed_fraction,
        shuffle=shuffle_fixed, seed=seed, rng=rng)

    # setup kd mode
    if mode == 'kd':
        n_fixed = fixed_pattern.shape[0]
        colocalize_fraction = fixed_fraction * n_fixed / n_colocalize
        if colocalize_fraction > 1:
            raise ValueError(
                f"Calculated colocalization fraction {colocalize_fraction} "
                + f"in 'kd' mode should not be >1. Please adjust arguments "
                + f"fixed_pattern, n_colocalize and fixed_fraction so that "
                + f"fixed_pattern.shape[0] * n_fixed / n_colocalize < 1.")
        mode = 'max1'
    
    # generate coclustered
    coclust = cocluster_region(
        clusters=fixed, N=n_colocalize, mode=mode,
        p_cluster=colocalize_fraction,
        region=region, region_id=region_id, region_coords=region_coords,
        max_dist=max_dist, exclusion=None,
        metric='euclidean', shuffle=shuffle_region, seed=seed, rng=rng)

    return coclust


#
# Common
#

def get_region_coords(
        region=None, region_id=None, region_coords=None, shuffle=True,
        seed=None, rng=None):
    """Extract coordinates of a region.

    If args region and region_id are specified, all points of image
    (arg) region that have value (arg) region_id are selected and their
    coordinates are extracted. If arg shuffle is True, the selected
    points are randomly shuffled. If arg region_coords is given and
    shuffle is True, region_coords are shuffled, which also changes
    the value of arg region_coords. 

    Alternatively, if args region and region_id are None, coordinates
    specified by (arg) region_coords are used. In this case they are
    only shuffled.
    
    Arguments:
      - region: (ndarray or pyto.core.Image) label image
      - region_id: id of the region in the label image (region)
      - region_coords: (ndarray, n_region_coords x n_dim) coordinates of all 
      points of a region, used if args region and region_id are not specified
      - shuffle: flag indicating if extracted coordinates should be randomly
      shuffled (default True)
      - rng: random number generator for shuffling, if None a new one
      is created
      - seed: random number generator seed for shuffling, used in case
      a new random number generator is created here
 
    Returns: (n_points x n_dim int ndarray) all coordinates of the region. 
    """

    # figure out how the region is specified
    if (region is not None) and (region_id is not None):

        if isinstance(region, pyto.core.Image):
            region = region.data
        region_coords = np.stack(np.nonzero(region==region_id), axis=1)

    elif region_coords is not None:
        region_coords = np.asarray(region_coords)

    else:
        raise ValueError(
            "A region has to be specified by eather an image (args region "
            + "and region_id args), or coordinates of all region points "
            + "(arg region_coords).")

    # shuffle region coords
    if shuffle:
        region_coords = region_coords.copy()
        if rng is None:
            rng = default_rng(seed=seed)
        rng.shuffle(region_coords)

    return region_coords

def select_by_region(
        pattern, region=None, region_id=None, region_coords=None, 
        fraction=1, shuffle=True, #shuffle_region=True,
        seed=None, rng=None):
    """Selects a fraction of a pattern on a region.

    The selection proceeds as follows:
      - Select (arg) fraction of (arg) pattern points, where
      np.round(fraction * n_points) are selected.
      - Further select those that belong to the region (specified by
      args region and region_id, or by region_coords).

    If shuffle is True (default), the pattern points are first shuffled and 
    then they are selected from the beginning. This ensures that the selected
    points are not biased by their order in pattern.
    
    If suffle is False, pattern points are selected (by fraction) from
    the beginning, in the order of pattern_region. This ensures that
    given the same fraction, the same points are selected. Depending on
    the usage, it may be important to make sure pattern contains points
    in a random order.

    If both args region and region_coords are None, selection by region
    is omitted.
    
    Arguments:
      - pattern: (ndarray n_points x n_dim) point pattern
      - region: (ndarray or pyto.core.Image) label image
      - region_id: id of the region in the label image (region)
      - region_coords: (ndarray, n_region_coords x n_dim) coordinates of all 
      points of a region, used if args region and region_id are not specified
      - fraction: fraction of the point pattern that is selected (default 1)
      - shuffle: flag indicating if the pattern coordinates are
      randomly shuffled before the selection (recommended unless the order
      of selected points has to be preseved, default True)
      - rng: random number generator for shuffling, if None a new one
      is created
      - seed: random number generator seed for shuffling, used in case
      a new random number generator is created here

    Return: (ndarray n_points x n_dim) selected points
    """

    # figure out region points
    if region_coords is None:
        if region is not None:
            region_coords = get_region_coords(
                region=region, region_id=region_id, shuffle=False,
                seed=seed, rng=rng)

    # select fraction
    if shuffle:
        pattern = pattern.copy()
        if rng is None:
            rng = default_rng(seed=seed)
        rng.shuffle(pattern)
    split_ind = int(np.round(pattern.shape[0] * fraction))
    selected, _ = np.split(pattern, [split_ind], axis=0)

    # select those in region
    if region_coords is not None:
        selected = np.asarray(
            [po for po in selected
             if np.array([(po == reg).all() for reg in region_coords]).any()])
    
    return selected

def get_rectangle_coords(rectangle):
    """Returns all coordinates of a rectangle.

    Argument:
      - rectangle: rectangle defined as
      [[x_min, y_min, ...], [x_max, y_max, ...]

    Returns ndarray (n_points x n_dim) coordinates of the rectangle points 
    """
    result = np.asarray(list(
        itertools.product(
            *[list(range(low, high)) for low, high in zip(*rectangle)])))
    return result

#
# Projected
#

def project(
        points, region_coords=None, region=None, region_id=None,
        project_mode='closest', line_projection=None,
        line_project_angles=None, line_project_distance=None,
        not_found=[-1,-1,-1],
        shuffle=True, seed=None, rng=None):
    """Projects multiple points on a region.

    The projection can be done in two ways:
      - project_mode='closest': Points are projected to their closest
      region points
      - project_mode='line': Points are projected along a specified line
      and the projected points are obtained at the line - region
      intersection point that is the closest to the points.

    See the docs in pyto.spatial.LineProjection, including project()
    and __init__() for more info about the 'line' mode.

    In line 'line' mode, region coordinates are expected to be passed
    to this function as args region_coords, or region and region_id.
    The analogous attributes of (arg) line_projection object are ignored,
    contrary to the usage of LineProjection alone.

    Args line_projection, line_project_angles and line_project_distance
    are used only in 'line' projection mode. The meaning of arg angles
    and the way line projection is performed is determined by the 
    attributes of line_projection object. The line projection object has 
    to have attribute intersect_mode set to 'first'.

    If in the line mode a projection is not found, coordinates specified
    by arg not_found are entered. This should be something like
    [-1, -1, -1] (default).

    Arguments:
      - points: (ndarray n_points x 3) coordinates of points from which
      the line projection is made
      - region_coords: (ndarray n_region_points x 3) coordinates of all
      region points (if None, region and region_id has to be specified)
      - region: (pyto.core.Image, ndarray, or file path) region image 
      onto which points are projected (used only if region_coords is None)
      - regin_id: id of the region of interest in the region image  
      (used only if region_coords is None)
      - project_mode: projection mode, 'closest' or 'line'
      - line_projection: (pyto.spatial.LineProjection) contains 
      line projection parameters
      - line_project_angles: 3 Euler angles, or just relion tilt and
      psi angles as specified in relion refinement star files
      - line_projection_distance: (single number or an array) projection 
      distance(s) [pixel], all of these are used for each point.
      - shuffle, seed, rng: indicate if and how region point are 
    randomized (see get_region_coords())

    Returns (ndarray n_points x 3, int) Coordinates of the projected points,
    or [-1, -1, -1] for each point for which projection could not be 
    determined.
    """

    # deal with the no points case
    if points.shape[0] == 0:
        projected = np.array([]).reshape(0, 3)
        return projected

    # figure out region points
    if region_coords is None:
        if isinstance(region, str):
            region = pyto.segmentation.Labels.read(
                file=region, memmap=True)
        region_coords = get_region_coords(
            region=region, region_id=region_id, shuffle=shuffle,
            seed=seed, rng=rng)

    if project_mode == 'closest':
        dist = cdist(points, region_coords)
        try:
            min_inds = dist.argmin(axis=1)
        except ValueError as e:
            raise ValueError(
                "Projection failed because region does not exist") from e
        projected = region_coords[min_inds]

    elif project_mode == 'line':
        n_points = points.shape[0]
        projected = -np.ones(shape=(n_points, 3), dtype=int)
        for point_ind in range(n_points):
            poi = points[point_ind, :]
            ang = line_project_angles[point_ind, :]
            line_projection.region_coords = region_coords
            proj = line_projection.project(
                angles=ang, point=poi, distance=line_project_distance)
            try:
                projected[point_ind, :] = proj
            except TypeError:
                projected[point_ind, :] = not_found

                
    return projected
 
