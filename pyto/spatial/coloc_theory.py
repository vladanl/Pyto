"""
Functions for analytical calculations of some of the colocalization properties.

# Author: Vladan Lucic
# $Id:"
"""

__version__ = "$Revision$"


import numpy as np
import scipy as sp
import pandas as pd 

import pyto


def coloc_random(
        distance, N, A, A_over, ndim=2, grid=False, d_factor=1,
        hood_area_factor=1, over_factor=0):

    # figure out distance
    d_single = False
    if not isinstance(distance, (list, np.ndarray, tuple)):
        distance = [distance]
        d_single = True

    # neighborhood area dependence on dimensionality and corrections
    if (ndim == 2) and grid:
        hood = np.asarray(grid_area_circle(radii=distance))
    else:
        hood = hood_area_factor * (4 / np.pi)\
            * (np.pi * np.asarray(d_factor * distance) / 2)**ndim

    # n coloc
    p_1plus = np.asarray([
        1 - (1 - hood / A_x)**N_x
        for N_x, A_x in zip(N[1:], A[1:])])
    p_1plus_all_coloc = np.multiply.reduce(p_1plus, axis=0)
    n_coloc = (p_1plus_all_coloc * N[0] * A_over
               * (1 + over_factor * distance)/ A[0])

    if d_single:
        n_coloc = n_coloc[0]

    return n_coloc    
    
def coloc_random_old(distance, N, A, A_over, ndim=2, grid=False, d_factor=1,
                 area_factor=1):
    """Calculates number of random 2- or 3- (or higher) colocalizations.
 
    Does not take into account border effects, nor exclusion distance 
    between points.

    Arguments:
      - distance: (single number or a list (ndarray) of numbers) 
      colocalization distance (in pixels)
      - N: (list or nparray of length n_sets) Number of points in each set
      - A: (list or nparray of length n_sets) Area where the points area
      distributed for each set
      - A_over: (list or nparray of length n_sets) Overlaps between the 
      above areas, where A_over[0] is the complete overlap (A{0] and A[1]
      and ...) and A_over[i] is th overlap between A[0] and A[i]. If 
      n_sets is 2, it can be specified as a single number
      - n_dim: NUmber of spatial dimensions, can be 1 or 2
      - grid (experimental): calculate area on the grid

    Returns number of colocalizations, in the same format as arg distance 
    """

    # figure out distance
    d_single = False
    if not isinstance(distance, (list, np.ndarray, tuple)):
        distance = [distance]
        d_single = True

    # setup for different n dim
    if ndim == 1:
        d_area = 2 * np.array(distance)
    elif ndim == 2:
        if grid:
            d_area = np.asarray(grid_area_circle(radii=distance))
        else:
            d_area = area_factor * np.pi * np.array(distance * d_factor)**2
    else:
        d_area = (4 / np.pi) * (np.pi * np.asarray(distance) / 2)**ndim
        #ValueError("Argument ndim has to be 1 or 2")

    # check overlap arg
    if (len(N) == 2) and not isinstance(A_over, (list, tuple, np.ndarray)):
        A_over = [A_over, A_over]
        
    # probability that at least one element each of sets 1 2, ... is in the
    # distance neghborhood of an element of set 0 that is in the complete
    # overlap region
    p_1plus_in_d0_all = np.asarray([
        1 - (1 - d_area / A_over_x)**(N_x * A_over_x / A_x)
        for N_x, A_x, A_over_x in zip(N[1:], A[1:], A_over[1:])])
    p_1plus_in_d0 = np.multiply.reduce(p_1plus_in_d0_all, axis=0)

    # take into account probability that an element of set 0 is in the
    # complete overlap
    n_coloc = p_1plus_in_d0 * N[0] * A_over[0] / A[0]

    if d_single:
        n_coloc = n_coloc[0]

    return n_coloc

def area_2circles(radii, center_dist):
    """
    Calculates the total area of two overlaping circles analytically.
    
    The caluclation is done for multiple radii. The two circles have to be 
    of the same size.

    Arguments:
      - radius: (list) circle radii
      - center_dist: distance between centers of the two circles

    Returns list of areas
    """
    
    area = []
    for radius in radii:
    
        separation = center_dist / (2 * radius)
        if separation > 1:
            curr_area = 2 * np.pi * radius**2
        else:
            alpha = np.arccos(separation)
            curr_area = 2 * (
                (np.pi - alpha) * radius**2 
                + center_dist * np.sqrt(radius**2 - center_dist**2 / 4))
        area.append(curr_area)
        
    return area

def grid_area_circle(radii, spacing=1, border=False):
    """Calculates surface area of a circle on a flat 2D rectangular grid.

    Circle radii have to be given in the same spatial units as spacing.

    Arguments:
      - radii: array or radii for which the area is calculated
      - spacing: grid spacing (the same as pixel size)
      - border: flag indicating if points on border are calculated
      
    Return: ndarray of areas in the spatial units defined by spacing.
    """

    area = []
    for dist in radii:

        # get spacing points centered on 0
        x_plus = np.arange(0, dist+spacing, spacing)
        x_range = np.concatenate([np.arange(-x_plus[-1], 0, spacing), x_plus])

        # calculate distances to origin
        xm, ym = np.meshgrid(x_range, x_range)
        rho_squared = xm**2 + ym**2
        
        # count spacing points 
        if border:
            n_points = (rho_squared <= dist**2).sum()
            #print(rho_squared)
        else:
            n_points = (rho_squared < dist**2).sum()

        # convert to area
        area.append(n_points)
        
    area = np.array(area) * spacing**2
    return area

def grid_area_multi_circles(radii, centers, spacing=1, border=False):
    """
    Calculates area covered by multiple circles on a flat 2D square grid.
    
    Extends grid_area_circle().
    
    Centers are given in pixles, while radii and spacing in spatial units 
    (like nm).

    For example, if spacing is 1.5 and radius is 1.2, they are both understood 
    in the same units (such as nm), so the distance between two centers 
    [[0, 0], [1, 0]] is 1 pixel = 1.5 nm. The returned ara will be in nm^2.

    The reason for using both pixels and spatial units is that centers are 
    typically particle locations determined directly from images in pixels, 
    while radii are determined based on the real (inter-)particle distances.

    Arguments:
      - radii: radii: list (array) of circle radii, in spatial units defined by 
      arg spacing
      - centers: circle center coordinates, shape (N, 2) where N is the 
      number of circles, in pixels
      - spacing: grid spacing (the same as pixel size) in spatial units
      - border: flag indicating whether border is incluced in the calulcated area
      
    Returns (ndarray) calculated area (in units defined by arg spacing) for 
    all distances.
    """

    # to include
    #pacing = 1.
    
    # centers to int
    cent = np.asarray(centers).round().astype(int)
    x_centers = cent[:, 0]
    y_centers = cent[:, 1]
    
    # calculate area in pixels
    area = []
    for radius in np.asarray(radii) / spacing:
        
        # make coordinate grid
        x_plus = np.arange(
            np.min(x_centers), np.max(x_centers) + radius + 1, 1)
        x_minus = np.flip(
            np.arange(np.min(x_centers), np.min(x_centers) - radius - 1, -1))[:-1]
        x_range = np.concatenate([x_minus, x_plus])
        y_plus = np.arange(
            np.min(y_centers), np.max(y_centers) + radius + 1, 1)
        y_minus = np.flip(
            np.arange(np.min(y_centers), np.min(y_centers) - radius - 1, -1))[:-1]
        y_range = np.concatenate([y_minus, y_plus])
        xm, ym = np.meshgrid(x_range, y_range)
        
        # extend the grid, one for each circle
        n_circles = len(x_centers)
        xm_ext = np.repeat(xm[..., np.newaxis], repeats=n_circles, axis=2)
        ym_ext = np.repeat(ym[..., np.newaxis], repeats=n_circles, axis=2)
        
        # calculate distance to center for all circles
        x_diff = xm_ext - np.expand_dims(x_centers, (0, 1))
        y_diff = ym_ext - np.expand_dims(y_centers, (0, 1))
        rho = x_diff**2 + y_diff**2

        # count elements that are within at least one circle
        if border:
            inside = (rho <= radius**2)
        else:
            inside = (rho < radius**2)   
        curr_area = np.logical_or.reduce(inside, axis=2).sum()
        
        area.append(curr_area)

    # return area in the spatial units specified by spacing 
    return np.array(area) * spacing**2


