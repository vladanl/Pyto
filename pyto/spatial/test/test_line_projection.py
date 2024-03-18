"""
Tests module line_projection

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import os
import unittest

import numpy as np
import numpy.testing as np_test
#import pandas as pd
#from pandas.testing import assert_frame_equal, assert_series_equal

from pyto.segmentation.labels import Labels
from pyto.spatial.test import common
from pyto.spatial.point_pattern import get_region_coords
from pyto.spatial.line_projection import LineProjection


class TestLineProjection(np_test.TestCase):
    """
    Tests line_projection module
    """

    def setUp(self):
        """
        """

        # y-slice region
        self.y_region_id = 2
        self.y_region_data = np.zeros((10, 10, 5), dtype=int)
        self.y_region_data[:, 4:7, 3] = self.y_region_id
        self.y_region = Labels(data=self.y_region_data)

        # thin diagonal region
        self.diag_region_id = 3
        self.diag_region_data = np.zeros((7, 7, 5), dtype=int)
        self.diag_region_data[5, 0, 2] = self.diag_region_id
        self.diag_region_data[4, 1, 2] = self.diag_region_id
        self.diag_region_data[3, 2, 2] = self.diag_region_id
        self.diag_region_data[2, 3, 2] = self.diag_region_id
        self.diag_region_data[1, 4, 2] = self.diag_region_id

    def test_find_spherical(self):
        """Tests find_spherical()
        """

        # relion
        lp = LineProjection(relion=True)
        relion_angles = [30, 140]
        desired = [30, 40]
        actual = lp.find_spherical(angles=relion_angles)
        np_test.assert_array_almost_equal(actual, desired)
        relion_angles = [20, 40, 70]
        desired = [40, 110]
        actual = lp.find_spherical(angles=relion_angles)
        np_test.assert_array_almost_equal(actual, desired)
        relion_angles = [-20, -40, 370]
        desired = [40, -10]
        actual = lp.find_spherical(angles=relion_angles)
        np_test.assert_array_almost_equal(actual, desired)

        # relion reverse, euler_range='0_2pi'
        lp = LineProjection(relion=True, reverse=True, euler_range='0_2pi')
        relion_angles = [-30, 240]
        desired = [150, 300]
        actual = lp.find_spherical(angles=relion_angles)
        np_test.assert_array_almost_equal(actual, desired)
        relion_angles = [20, 200, 210]
        desired = [20, 330]
        actual = lp.find_spherical(angles=relion_angles)
        np_test.assert_array_almost_equal(actual, desired)
       
        # relion reverse, euler_range='-pi_pi'
        lp = LineProjection(relion=True, reverse=True, euler_range='-pi_pi')
        relion_angles = [-30, 240]
        desired = [150, -60]
        actual = lp.find_spherical(angles=relion_angles)
        np_test.assert_array_almost_equal(actual, desired)
        relion_angles = [20, 200, 210]
        desired = [20, -30]
        actual = lp.find_spherical(angles=relion_angles)
        np_test.assert_array_almost_equal(actual, desired)
       
        # 'zyz_ex_active'
        lp = LineProjection(
            relion=False, degree=False, euler_mode='zyz_ex_active')
        angles = [0.2, 0.3, 0.4]
        desired = [0.3, -0.2 + np.pi]
        actual = lp.find_spherical(angles=angles)
        np_test.assert_array_almost_equal(actual, desired)
        
        # 'zyz_ex_active', reverse
        lp = LineProjection(
            relion=False, degree=False, euler_mode='zyz_ex_active',
            reverse=True)
        angles = [0.2, 0.3, 0.4]
        desired = [np.pi - 0.3, -0.2]
        actual = lp.find_spherical(angles=angles)
        np_test.assert_array_almost_equal(actual, desired)
        
         # 'zyz_ex_active', reverse
        lp = LineProjection(
            relion=False, degree=False, euler_mode='zyz_ex_active',
            reverse=True, euler_range='0_2pi')
        angles = [0.2, 0.3, 0.4]
        desired = [np.pi - 0.3, -0.2 + 2 * np.pi]
        actual = lp.find_spherical(angles=angles)
        np_test.assert_array_almost_equal(actual, desired)

    def test_project_along_line(self):
        """Tests project_along_line()
        """

        # one point
        lp = LineProjection(degree=False)
        theta, phi = (np.pi/2., 0)
        point = [0, 0, 3]
        distance = 2
        desired = [2, 0, 3]
        actual = lp.project_along_line(
            theta=theta, phi=phi, distance=distance, point=point)
        np_test.assert_almost_equal(actual, desired)
        
        # simple
        lp = LineProjection(degree=True)
        theta, phi = (90, 90)
        distance=np.linspace(1, 5, 5)
        desired = np.zeros((5, 3))
        desired[:, 1] = distance
        actual = lp.project_along_line(
            theta=theta, phi=phi, distance=distance)
        np_test.assert_array_almost_equal(actual, desired)

        # general
        lp = LineProjection(degree=False)
        theta, phi = (-0.3, 0.6)
        point = [1, 2, 3]
        distance = np.linspace(2, 4, 5)
        desired = [
            np.array([np.cos(phi) * np.sin(theta),
                      np.sin(phi) * np.sin(theta),
                      np.cos(theta)]) * dist
            + point
            for dist in distance]
        actual = lp.project_along_line(
            theta=theta, phi=phi, distance=distance, point=point)
        np_test.assert_array_almost_equal(actual, desired)

    def test_get_grid_points(self):
        """Tests get_grid_points
        """

        # mode nearest
        lp = LineProjection(grid_mode='nearest')
        points = np.array(
            [[1.2, 3.6, 4.7],
             [1.3, 3.7, 4.8],
             [11.6, 13.2, 14],
             [11.8, 13.3, 14.4]])
        desired = np.array([[1, 4, 5], [12, 13, 14]])
        actual = lp.get_grid_points(line_points=points)
        np_test.assert_array_equal(actual, desired)

        # mode unit
        lp = LineProjection(grid_mode='unit')
        points = np.array(
            [[1.2, 3.6, 4.7],
             [1.8, 3.4, 4.7],
             [11.6, 13.2, 14],
             [11.4, 13.6, 14.8]])
        desired = np.array(
            [[[1, 3, 4], [1, 3, 5], [1, 4, 4], [1, 4, 5],
              [2, 3, 4], [2, 3, 5], [2, 4, 4], [2, 4, 5]],
             [[11, 13, 14], [11, 13, 15], [11, 14, 14], [11, 14, 15],
              [12, 13, 14], [12, 13, 15], [12, 14, 14], [12, 14, 15]]])
        actual = lp.get_grid_points(line_points=points)
        np_test.assert_array_equal(actual, desired)

    def test_intersect_line_region(self):
        """Tests intersect_line_region
        """

        # flat region, mode first, nearest
        lp = LineProjection(
            region=self.y_region_data, region_id=self.y_region_id,
            grid_mode='nearest', intersect_mode='first')
        points = np.array(
            [[1, 2, 3], [1, 3, 3], [1, 4, 3], [1, 5, 3], [1, 6, 3]])
        desired = [1, 4, 3]
        actual = lp.intersect_line_region(grid_points=points)
        np_test.assert_array_equal(actual, desired)
       
        # flat region, mode first, nearest, no projected
        lp = LineProjection(
            region=self.y_region_data, region_id=self.y_region_id,
            grid_mode='nearest', intersect_mode='first')
        points = np.array([[1, 2, 3], [1, 3, 3]])
        #desired = [1, 4, 3]
        actual = lp.intersect_line_region(grid_points=points)
        np_test.assert_equal(actual is None, True)
       
        # flat region, mode first, unit
        lp = LineProjection(
            region=self.y_region, region_id=self.y_region_id,
            grid_mode='unit', intersect_mode='first')
        points = np.array(
            [[[1, 2, 3], [1, 2, 4]], [[1, 3, 3], [1, 3, 4]],
             [[1, 4, 3], [2, 4, 3]], [[1, 5, 3], [1, 5, 3]],
             [[1, 6, 3], [1, 6, 3]]])
        desired = [1, 4, 3]
        actual = lp.intersect_line_region(grid_points=points)
        np_test.assert_array_equal(actual, desired)
       
        # flat region, mode all, nearest
        lp = LineProjection(
            region=self.y_region, region_id=self.y_region_id,
            grid_mode='nearest', intersect_mode='all')
        points = np.array(
            [[1, 2, 3], [1, 3, 3], [1, 4, 3], [1, 5, 3], [1, 6, 3]])
        desired = [[1, 4, 3], [1, 5, 3], [1, 6, 3]]
        actual = lp.intersect_line_region(grid_points=points)
        np_test.assert_array_equal(actual, desired)
       
        # flat region, mode all, nearest, no projection
        lp = LineProjection(
            region=self.y_region, region_id=self.y_region_id,
            grid_mode='nearest', intersect_mode='all')
        points = np.array([[1, 2, 3], [1, 3, 3]])
        desired = []
        actual = lp.intersect_line_region(grid_points=points)
        np_test.assert_array_equal(actual, desired)
       
        # flat region, mode all, unit
        lp = LineProjection(
            region=self.y_region_data, region_id=self.y_region_id,
            grid_mode='unit', intersect_mode='all')
        points = np.array(
            [[[1, 2, 3], [1, 2, 4]], [[1, 3, 3], [1, 3, 4]],
             [[1, 4, 3], [2, 4, 3]], [[1, 5, 2], [1, 5, 3]],
             [[2, 6, 3], [1, 6, 3]], [[1, 7, 3], [1, 7, 4]]])
        desired = [[1, 4, 3], [1, 5, 3], [2, 6, 3]]
        actual = lp.intersect_line_region(grid_points=points)
        np_test.assert_array_equal(actual, desired)

        # thin diagonal region, mode first, nearest
        lp = LineProjection(
            region=self.diag_region_data, region_id=self.diag_region_id,
            grid_mode='nearest', intersect_mode='first')
        points = np.array(
            [[0, 0, 2], [1, 1, 2], [2, 2, 2], [3, 3, 2]])
        desired = []
        actual = lp.intersect_line_region(grid_points=points)
        np_test.assert_array_equal(actual, desired)
       
        # thin diagonal region, mode all, nearest
        lp = LineProjection(
            region=self.diag_region_data, region_id=self.diag_region_id,
            grid_mode='nearest', intersect_mode='all')
        points = np.array(
            [[0, 0, 2], [1, 1, 2], [2, 2, 2], [3, 3, 2]])
        desired = np.array([]) 
        actual = lp.intersect_line_region(grid_points=points)
        np_test.assert_array_equal(actual, desired)
       
        # thin diagonal region, mode first, unit
        lp = LineProjection(
            region=self.diag_region_data, region_id=self.diag_region_id,
            grid_mode='unit', intersect_mode='first')
        points = np.array(
            [[[0, 0, 2], [0, 1, 2]], [[1, 1, 2], [1, 2, 2]],
             [[2, 2, 2], [2, 3, 2]], [[3, 3, 2], [3, 4, 2]]])
        desired = [2, 3, 2]
        actual = lp.intersect_line_region(grid_points=points)
        np_test.assert_array_equal(actual, desired)

    def test_project(self):
        """Tests project()
        """

        # flat region
        lp = LineProjection(
            region=self.y_region, region_id=self.y_region_id, 
            relion=True, reverse=False,
            grid_mode='nearest', intersect_mode='first')
        actual = lp.project(
            angles=[-90, -90], distance=np.linspace(1, 5, 7), point=[1, 0, 3])
        np_test.assert_array_equal(actual, [1, 4, 3])

        # diagonal region, 'nearest'
        lp = LineProjection(
            region=self.diag_region_data, region_id=self.diag_region_id, 
            relion=True, reverse=False,
            grid_mode='nearest', intersect_mode='first')
        actual = lp.project(
            angles=[-90, -45], distance=np.linspace(1, 5, 9), point=[0, 0, 2])
        np_test.assert_array_equal(actual, [])

        # diagonal region, 'unit'
        lp = LineProjection(
            region=self.diag_region_data, region_id=self.diag_region_id, 
            relion=True, reverse=False,
            grid_mode='unit', intersect_mode='first')
        actual = lp.project(
            angles=[-90, -45], distance=np.linspace(1, 5, 9), point=[0, 0, 2])
        np_test.assert_array_equal(actual, [2, 3, 2])

        # diagonal region, 'unit', region coords
        region_coords = get_region_coords(
            region=self.diag_region_data, region_id=self.diag_region_id,
            shuffle=False)
        lp = LineProjection(
            region_coords=region_coords, relion=True, reverse=False,
            grid_mode='unit', intersect_mode='first')
        actual = lp.project(
            angles=[-90, -45], distance=np.linspace(1, 5, 9), point=[0, 0, 2])
        np_test.assert_array_equal(actual, [2, 3, 2])

        
      
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestLineProjection)
    unittest.TextTestRunner(verbosity=2).run(suite)
