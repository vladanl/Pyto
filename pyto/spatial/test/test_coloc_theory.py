"""

Tests module coloc_theory

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import os
import unittest

import numpy as np
import numpy.testing as np_test
import pandas as pd

from pyto.spatial.test import common
import pyto.spatial.coloc_theory as col_theory

class TestColocTheory(np_test.TestCase):
    """
    Tests coloc_theory module
    """

    def setUp(self):
        """
        """
        pass
          
    def test_area_2circles(self):
        """
        Tests area_2circles()
        """
    
        # center_dist = 2 * radius
        radii = np.array([3, 8])
        desired = 2 * np.pi * radii**2
        actual = col_theory.area_2circles(
            radii=radii, center_dist=2*radii.max())
        np_test.assert_array_almost_equal(actual, desired)

        # center dist = 0
        desired = np.pi * radii**2
        actual = col_theory.area_2circles(radii=radii, center_dist=0)
        np_test.assert_array_almost_equal(actual, desired)

        # center_dist = radius
        for radius in radii:
            desired = (4 * np.pi / 3 + np.sqrt(3)) * radius**2
            actual = col_theory.area_2circles(
                radii=[radius], center_dist=radius)
            np_test.assert_array_almost_equal(actual, desired)
        
    def test_grid_area_multi_circles(self):
        """
        Tests grid_area_multi_circles()
        """
    
        radii = [2, 4, 6, 8]
        spacing = 1.3
        one_circle = col_theory.grid_area_circle(
            radii=radii, spacing=1, border=False)
        one_circle_space = col_theory.grid_area_circle(
            radii=radii, spacing=spacing, border=False)
        
        # one circle
        obtained = col_theory.grid_area_multi_circles(
            radii=radii, centers=[[2, 3]], border=False)
        np_test.assert_array_almost_equal(obtained, one_circle)
        obtained = col_theory.grid_area_multi_circles(
            radii=radii, centers=[[2, 3]], spacing=spacing, border=False)
        np_test.assert_array_almost_equal(obtained, one_circle_space)
    
        # completely overlapping circles
        obtained = col_theory.grid_area_multi_circles(
            radii=radii, centers=[[2, 3], [2, 3], [2, 3]], border=False)
        np_test.assert_array_almost_equal(obtained, one_circle)
        obtained = col_theory.grid_area_multi_circles(
            radii=radii, centers=[[2, 3], [2, 3], [2, 3]], spacing=spacing,
            border=False)
        np_test.assert_array_almost_equal(obtained, one_circle_space)
     
        # non-overlapping circles
        obtained = col_theory.grid_area_multi_circles(
            radii=radii, centers=[[-2, -4], [8, 20]], border=False)
        np_test.assert_array_almost_equal(obtained,  2 * np.array(one_circle))
        obtained = col_theory.grid_area_multi_circles(
            radii=radii, centers=[[-2, -4], [8, 20]], spacing=spacing,
            border=False)
        np_test.assert_array_almost_equal(
            obtained, 2 * np.array(one_circle_space))
   
        # two partially overlapping circles
        centers = [[-2, 1], [1, 5]]
        center_dist = 5
        desired = col_theory.area_2circles(radii, center_dist)
        obtained = col_theory.grid_area_multi_circles(
            radii=radii, centers=centers, spacing=1, border=False)
        #print(obtained, desired)
        

if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestColocTheory)
    unittest.TextTestRunner(verbosity=2).run(suite)
