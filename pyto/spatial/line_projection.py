"""
Project a point along a line on a region.

# Author: Vladan Lucic 
# $Id$
"""

__version__ = "$Revision$"


import itertools

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist
import pandas as pd

import pyto
from pyto.geometry.rigid_3d import Rigid3D
from .point_pattern import get_region_coords


class LineProjection:
    """Class for projecting a point along a line on a region.

    Typical usage scenarions:

    1) Project by line determned by relion refinement angles:

      lp = LineProjection(
              region=region_image_or path, region_id=region_id,
              relion=True, reverse=True_to_reverse_line_direction, 
              grid_mode='unit', intersect_mode='first')
      projected_points = lp.project(
              point=origin_point, angles=relion_angles,
              distance=range(min_distance, max_distance))
    where:
      - angles = [rlnAngleRot, rlnAngleTilt, rlnAnglePsi] or
        angles = [rlnAngleTilt, rlnAnglePsi]
    (the same because rlnAngleRot is irrelevant here)

    2) Project by line defined by spherical angles:

      lp = LineProjection(
              region=region_image_or path, region_id=region_id,
              relion=False, spherical=True, 
              reverse=True_to_reverse_line_direction, 
              degree=True_for_deg_False_for_rad,
              grid_mode='unit', intersect_mode='first')
      projected_points = lp.project(
              point=origin_point, angles=[phi, theta],
              distance=range(min_distance, max_distance))
 
    For >1 pixel thick regions without holes grid_mode='nearest' will
    be more efficient.
    """

    @classmethod
    def project_mps_to_rm(
            cls, mps, region_id, coord_cols, out_cols,
            distance, reverse=False, grid_mode='unit',
            region_col=None, angle_cols=['rlnAngleTilt', 'rlnAnglePsi'],
            not_found=[-1, -1, -1]):
        """Projects particles on regions for multiple tomos.

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
            cls.project_mps_one_to_rm(
                particles=part_one, region=region_path, region_id=region_id,
                coord_cols=coord_cols, angle_cols=angle_cols, out_cols=out_cols,
                distance=distance, reverse=reverse, grid_mode=grid_mode,
                not_found=not_found)
            part_list.append(part_one)

        mps.particles = pd.concat(part_list)
        
    @classmethod
    def project_mps_one_to_rm(
            cls, particles, region, region_id, coord_cols, angle_cols, out_cols,
            distance, reverse=False, grid_mode='unit', not_found=[-1,-1,-1]):
        """Projects particles from MultiParticleSets on a region for one tomo.

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
        #n_points = points.shape[0]
        n_points = particles.shape[0]
        lp = cls(
            region_coords=region_coords, relion=True, reverse=reverse,
            grid_mode=grid_mode, intersect_mode='first', not_found=not_found)
        particles[out_cols] = particles.apply(
            lambda x: lp.project(
                point=x[coord_cols].to_numpy(),
                angles=x[angle_cols].to_numpy(), distance=distance),
            axis=1, result_type='expand')
        
        #if n_points > 0:
            
            
        #    projected = -np.ones(shape=(n_points, 3), dtype=int)
        #    for point_ind in range(n_points):
        #        poi = points[point_ind, :]
        #        ang = angles[point_ind, :]
        #        proj = lp.project(
        #            angles=ang, point=poi, distance=distance)
                #try:
        #        projected[point_ind, :] = proj            
                #except TypeError:
                #    projected[point_ind, :] = not_found
                   
        #else:
        #    projected = np.array([]).reshape(0, 3)

        # add to particles
        #particles[out_cols] = projected
        if particles[out_cols].isna().any(axis=None):
            particles[out_cols] = particles[out_cols].astype('Int64')
       
    def __init__(
            self, region=None, region_id=None, region_coords=None,
            relion=False, spherical=False, euler_mode='zxz_ex_active',
            degree=False, euler_range='-pi_pi',
            reverse=False, grid_mode='nearest', intersect_mode='first',
            not_found=None):
        """Sets attributes from arguments.

        Arguments:
          - region: (pyto.core.Image or ndarray) region image, used to
          project on
          - region_id: id of the region of interest in region image
          - region_coords: (ndarray n_points x 3) coordinates of points
          that belong to the region of interest
          - relion: flag indicating if angles are understood as relion angles,
          same as euler_mode='zyz_in_active' and degree=True
          - spherical: flag indicating if angles are taken as simple 
          spherical angles, euler_mode is ignored
          - euler_mode: Euler angles mode (see 
          pyto.geometry.Rigid3D.make_r_euler() doc for available modes,
          disregarded if relion=True
          - degree: flag indication if Euler and the resulting angles are 
          in degrees (True) or radians (False),  disregarded if relion=True
          - reverse: flag indicating if the direction of the resulting 
          line should be reversed
          - grid_mode: 'nearest' or 'unit', determines conversion from 
          line points (floats) to the Cartesian grid (ints)
          - intersection_mode: 'first', 'all', determines whether the first
          or all intersetion points are returned
          - not_found: returned coordinates when in the intersection mode 
          'first' a projection is not found 
        """

        self.region = region
        self.region_id = region_id
        self.region_coords = region_coords
        self.relion = relion
        if self.relion:
            self.euler_mode = 'zyz_in_active'
            self.degree = True
        else:
            self.euler_mode = euler_mode
            self.degree = degree
        self.spherical = spherical
        if spherical and relion:
            raise ValueError(
                "Only one of args relion and spherical can be True.")
        self.euler_range = euler_range
        self.reverse = reverse
        self.grid_mode = grid_mode
        self.intersect_mode = intersect_mode
        self.not_found = not_found

    def project(self, point, angles, distance):
        """Projects a point along a line onto a region.

        In the standard case (self.intersection_mode = 'first'), does the 
        follwing steps:
          - determines spherical angles from arg angles that define a line
          for each point
          - projects from (arg) points along the lines at the specified 
          distance(s) (arg distance),
          - converts distance-defined points on each line to the 
          cartesian grid 
          - returns the first points at the intersection of lines with 
          the specified region 
        For more info see the docs for find_spherical(), project_along_line(),
        get_grid_points() and intersect_line_region() methods.

        If arg distance is an array and the intersection mode is 'first', 
        the elements of distance are expected to be sorted in the
        increasing order. In this way, even when the intersection of a line
        and a region comprises multiple points, the resulting projected point
        will be the one that is the closest to the original point (arg point).

        To detect the intersection between a line and a 1 voxel thin 
        region that is fully connected (without holes) according to
        connectivity=1 (in the scipy.ndimage sense, that is common faces) 
        the difference between neighboring elements of distance should 
        be <=1 and it is sufficient to have self.grid_mode = 'closest'.

        To detect the intersection between a line and a 1 voxel thin 
        region that is fully connected (without holes) according to
        connectivity=3 (in the scipy.ndimage sense, that is common corners) 
        the difference between neighboring elements of distance should 
        be <=1 and self.grid_mode = 'unit'.

        It is also possible to run this method when self.intersection_mode 
        is 'all', in which case all interaction points are returned.

        Requires attributes:
          - self.relion, or both self_euler_mode and self.degree
          - self.reverse
          - self.grid_mode
          - self.intersection_mode
          - self.region_coords, or both self.region and self.region_id

        Arguments:
          - point: (array of len 3, or 0) coordinates of a point from which
          the line projection is made, default 0, that is [0, 0, 0]
          - angles: Euler angles if spherical=False, otherwise spherical
          angles (phi, theta)
          - distance: (single number or an array) projection distance(s) 
          [pixel], default 1

        Returns (1d ndarray of length 3 self.intersection_mode = 'first', 
        or n_points x 3 ndarray if self.intersection_mode = 'all') 
        coordinates of the projected points. If no projected points are 
        found, returns None if intersect mode is 'first', and [] if 
        intersect mode is 'all'.
        """

        if self.spherical:
            phi, theta = angles
        else:
            theta, phi = self.find_spherical(angles=angles)
        line_points = self.project_along_line(
            theta=theta, phi=phi, distance=distance, point=point)
        if isinstance(self.grid_mode, (int, float)):
            line_points_on_grid = np.asarray(line_points, dtype=float)
        else:
            line_points_on_grid = self.get_grid_points(
                line_points=line_points)
        intersection = self.intersect_line_region(
            grid_points=line_points_on_grid)

        return intersection
        
    def find_spherical(self, angles):
        """Finds line direction from Euler angles

        In relion mode (relion=True), arg angles is taken to be active 
        (from particle to reference, contrary to relion docs) intrinsic 
        zyz Euler angles (arg euler mode is ignored). They can be 
        specified as:
          - [rot, tilt, psi]
          - [tilt, psi]  
        Returns sperical coordinates (theta, phi):
          - [tilt, pi - psi] if 0<tilt<pi and self.reverse is False
          - [-tilt, -psi] if -pi<tilt<0 and self.reverse is False
          - [pi - tilt, -psi] if 0<theta<pi and self.reverse is True
          - [pi + tilt, pi - psi] if -pi<theta<0 and self.reverse is True

        This can be seen because the following are equivalent:
          - Relion (intrinsic, active, zyz): [rot, tilt, psi]
          - extrinsic, active, zyz: [psi, tilt, rot]
          - extrinsic, passive, zyz: [-rot, -tilt, -psi]
        Because the vector that defines spherical angles can be obtained as 
        extrinsic, passive, zyz Euler transformation of a line along z axis 
        oriented in the +z direction:
          - Spherical: [not_defined, theta, phi]
        It follows (the following two are euqivalent): 
            theta = -tilt, phi = -psi
            theta = tilt, phi = pi - psi
        For reversed vector (self.reverse is True):
            theta = pi - tilt, phi = -psi
            theta = pi + tilt, phi = pi - psi        

        In the normal mode (self.relion=False), arg angles have to be in the 
        Euler mode specified by self.euler_mode in the order:
          - [phi, theta, psi]

        Requires attributes:
          - self.relion, or both self_euler_mode and self.degree
          - self.reverse

        Arguments:
          - angles: Euler angles in degrees (if self.degree is True), or
          in radians (if self.degree is False)
 
        Returns: (theta, phi) spherical angles of the line, in the same 
        units as angles (specified by arg degree)
        """

        # setup relion mode
        euler_init = np.zeros(3)
        if self.relion:
            if len(angles) == 2:
                euler_init[1:] = angles
            elif len(angles) == 3:
                euler_init = np.asarray(angles)
        else:
            euler_init = np.asarray(angles)

        # get normals and normalize
        if self.degree:
            euler_init = euler_init * np.pi / 180

        # reverse if needed, convert and normalize
        if self.reverse:
            euler_init = Rigid3D.reverse_euler(angles=euler_init, degree=False)
        euler_final = Rigid3D.convert_euler(
            angles=euler_init, init=self.euler_mode, final='zyz_ex_passive')
        euler_final = np.asarray(Rigid3D.normalize_euler(
            euler_final, range=self.euler_range))
        theta_psi = euler_final[1:]

        if self.degree:
            theta_psi = 180 * theta_psi / np.pi

        return theta_psi            

    def project_along_line(self, theta, phi, distance=1, point=0):
        """Find coordinates of a line at a given distance(s) from a point.

        More precisely, returns coordinates of one or more ponts that
        belong to the line defined by shperical coordinates (args theta
        and phi) and the origin point (arg point) and are at the 
        specified distance(s) (arg distance) from the origin point.

        The returned coordinates are in general not on the cartesian grid
        (dtype float).

        Arguments:
          - theta, psi: spherical angles that determine the line direction
          (in rad or degrees, depending on self.degree)
          - distance: (single number or an array) projection distance(s) 
          [pixel], default 1
          - point: (array of len 3, or 0) coordinates of a point from which
          the line projection is made, default 0, that is [0, 0, 0]

        Returns (array of shape 3, or (n_distances, 3), depending on the 
        form of arg distance, type float) coordinates of the projection line 
        coordinates for specified distances. 
        """

        # get rotation matrix
        angles = np.zeros(3)
        angles[1:] = [theta, phi]
        if self.degree:
            angles = np.pi * angles / 180
        euler_mode = 'zyz_ex_active'  # not needed in args
        q = Rigid3D.make_r_euler(angles=angles, mode=euler_mode)

        # transform
        unit_z = np.array([0, 0, 1.]) 
        if isinstance(distance, (list, tuple, np.ndarray)):
            displace = [
                Rigid3D(q=q, scale=dist).transform(
                    x=unit_z, d=point, xy_axes='point_dim')
                for dist in distance] 
        else:
            displace_transf = Rigid3D(q=q, scale=distance, d=point)
            displace = displace_transf.transform(
                x=unit_z, d=point, xy_axes='point_dim')

        return displace

    def get_grid_points(self, line_points):
        """Converts points on a line to coordinate on Cartesion grid.

        If self.grid_mode is 'nearest', selects the closest grid point 
        for each point of line_points.
        
        If self.grid_mode is 'unit', for each line point, selects an 
        8x3 array of coordinates, where the closest two grid coordinates 
        are taken in each dimension. For example, for point:
          [1.2, 4.6, 7]
        the resulting 8x3 array is:
          [[1, 4, 7], [1, 4, 8], [1, 5, 7], [1, 5, 8],
           [2, 4, 7], [2, 4, 8], [2, 5, 7], [2, 5, 8]]

        Only the unique points among the selected ones are returned. If
        self.grid_mode is 'unit' the unique 8-point arrays are returned. 

        Consequently, grid mode 'nearest' is expected to be much faster
        than 'unit'.

        Requires attribute:
          - self.grid_mode

        Argument:
          - line_points: (array n_points x 3, float) coordinates on
          a line

        Returns grid coordinates:
          - n_points x 3 if self.grid_mode is 'nearest'
          - n_points x 8 x 3 if self.grid_mode is 'unit'
        """

        line_points = np.asarray(line_points)
        
        if isinstance(self.grid_mode, str) and (self.grid_mode == 'nearest'):
            grid_points = line_points.round().astype(int)

        elif isinstance(self.grid_mode, str) and (self.grid_mode == 'unit'):
            flo = np.floor(line_points).astype(int)
            product_list = []
            for point in flo:
                prod = list(itertools.product(*zip(point, point+1)))
                product_list.append(np.asarray(prod))
            grid_points = np.stack(product_list, axis=0)

        elif isinstance(self.grid_mode, (int, float)):
            grid_points = np.asarray(line_points, dtype=float)
            
        else:
            raise ValueError(
                f"Sorry, grid mode {self.grid_mode} was not understood.")

        # keep only unique
        _, indices = np.unique(grid_points, return_index=True, axis=0)
        grid_points_unique = grid_points[indices]
            
        return grid_points_unique

    def intersect_line_region(self, grid_points):
        """Finds intrsection between points on cartesian grid and a region.

        The region can be defined by self.region_coords (array of region 
        coordinates). If self.region_coords, region is defined by 
        region image, in which case self.region and self.region_id have to
        be defined.

        If self.grid_mode is 'first', returns coordinates of the first
        point (in the order of arg grid_points) that belong to the region.

        If self.grid_mode is 'all', and self.grid_mode == 'nearest',
        returns coordinates of all off grid points that belong to the region.  

        If self.grid_mode is 'all', and self.grid_mode == 'unit', 
        only the first intersecting point among the 8 'unit' array is
        returned.

        Reguires attributes:
          - self.region_coords or both self.region and self.region_id
          - self.grid_mode
          - self.intersection_mode
          - self.not_found

        Argument:
          - grid points: coordinate of grid points, as returned by 
          get_grid_points(), that is n_points x 8, 3

        Returns (1d ndarray of length 3 if self.intersection_mode = 'first', 
        or n_points x 3 ndarray if self.intersection_mode = 'all') 
        coordinates of the projected points. If no projected points are 
        found, returns None if intersect mode is 'first', and [] if 
        intersect mode is 'all'.
        """

        # figure out region
        if self.region_coords is None:
            region_coords = get_region_coords(
                region=self.region, region_id=self.region_id, shuffle=False)
        else:
            region_coords = self.region_coords
 
        # intersect
        if self.intersect_mode == 'first':
            
            if (isinstance(self.grid_mode, str)
                and (self.grid_mode == 'nearest')):
                for one_point in grid_points:
                    intersect = np.logical_and.reduce(
                        one_point == region_coords, axis=1)
                    if intersect.any():
                        result = one_point
                        break
                else:
                    result = self.not_found
 
            elif (isinstance(self.grid_mode, str)
                  and (self.grid_mode == 'unit')):
                for point_ind, grid_ind in itertools.product(
                        range(grid_points.shape[0]),
                        range(grid_points.shape[1])):
                    one_point = grid_points[point_ind, grid_ind, :]
                    intersect = np.logical_and.reduce(
                        one_point == region_coords, axis=1)
                    if intersect.any():
                        result = one_point
                        break
                else:
                    result = self.not_found

            elif isinstance(self.grid_mode, (int, float)):
                dist = cdist(grid_points, region_coords)
                dist_cond = dist <= self.grid_mode
                hood_points = np.logical_or.reduce(dist_cond, axis=1)
                try:
                    first_ind = hood_points.nonzero()[0].min()
                except ValueError:
                    result = self.not_found
                else:
                    closest_reg_index = dist[first_ind, :].argmin()
                    result = region_coords[closest_reg_index, :]
                
        elif self.intersect_mode == 'all':

            if self.grid_mode == 'nearest':
                intersect_flags = []
                for one_point in grid_points:
                     intersect = np.logical_and.reduce(
                        one_point == region_coords, axis=1)
                     intersect_flags.append(intersect.any())
                result = grid_points[intersect_flags]
                if result.shape[0] == 0:
                    result = np.asarray([])
                
            elif self.grid_mode == 'unit':
                result = []
                for point_ind in range(grid_points.shape[0]):
                    for grid_ind in range(grid_points.shape[1]):
                        one_point = grid_points[point_ind, grid_ind, :]
                        intersect = np.logical_and.reduce(
                            one_point == region_coords, axis=1)
                        if intersect.any():
                            result.append(one_point)
                            break
                result = np.asarray(result)

        else:
            raise ValueError(
                f"Sorry, intersection mode {self.intersect_mode} was "
                + "not understood.")
            
        return result
