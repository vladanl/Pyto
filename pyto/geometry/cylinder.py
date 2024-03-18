"""
Makes cylinder in 3D

# Author: Vladan Lucic 
# $Id:"
"""

__version__ = "$Revision$"

import itertools

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist


class Cylinder:
    """3D cylinder
    """

    def __init__(self, z_min, z_max, rho, shape=None, axis_xy='center'):
        """Sets attributes from arguments.

        """

        self.z_min = z_min
        self.z_max = z_max
        self.rho = rho
        if isinstance(shape, (tuple, list, np.ndarray)):
            self.shape = shape
        else:
            self.shape = [shape, shape, shape]
        if isinstance(axis_xy, str):
            if axis_xy == 'center':
                self.axis_xy = [self.shape[0] // 2, self.shape[1] // 2]
        else:
            self.axis_xy = axis_xy    

    @classmethod
    def make_image(cls, z_min, z_max, rho, shape, axis_xy='center'):
        """Makes a cylinder image.

        """
        cyl = cls(
            z_min=z_min, z_max=z_max, rho=rho, shape=shape, axis_xy=axis_xy)
        cyl.make()
        return cyl.data
        
    def make(self):
        """Makes a 3D cylinder

        """

        # 2d coords
        indices_x = np.linspace(0, self.shape[0], self.shape[0]+1)
        indices_y = np.linspace(0, self.shape[1], self.shape[1]+1)
        z_slice_coords = (
            np.asarray(np.meshgrid(indices_x, indices_y, indexing='ij'))
            .reshape(2, -1)
            .transpose())

        # circle coords
        z_slice_dist = cdist(z_slice_coords, [self.axis_xy]).reshape(-1)
        z_slice_circle = z_slice_coords[z_slice_dist <= self.rho] 

        # cylinder coords
        n_z_points = np.round(self.z_max - self.z_min).astype(int) + 1
        z_coords = np.linspace(self.z_min, self.z_max, n_z_points)
        cyl_coords = np.array(
            [[slic[0], slic[1], z_co] for slic, z_co
             in itertools.product(z_slice_circle, z_coords)])
        self.coords = cyl_coords

        # make cylinder image
        cyl_coords = np.round(cyl_coords).astype(int)
        cyl = np.full(self.shape, False, dtype=bool)
        cyl[tuple(cyl_coords.transpose().tolist())] = True
        self.data = cyl
        
        
        
        
