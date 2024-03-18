"""
Tests class ConvexHullUtil

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"


import os
import warnings
import unittest

import numpy as np
import numpy.testing as np_test
import scipy as sp

import pyto
from pyto.geometry.convex_hull_util import ConvexHullUtil
from pyto.segmentation.labels import Labels

class TestConvexHullUtil(np_test.TestCase):
    """
    """

    def setUp(self):
        """Setup file paths
        """
        self.current_dir = os.path.dirname(os.path.realpath(__file__))
        self.intersect_path_in = os.path.join(
            self.current_dir, 'hull_intersect_in.mrc')
        self.intersect_path_out = os.path.join(
            self.current_dir, 'hull_intersect_out.mrc')

    def test_inside_hull(self):
        """Tests inside_hull()
        """

        # 2d, test hull and target points
        hull_points = np.array(
            [[1, 1], [5, 2], [7, 8], [1, 3], [2, 3], [3, 4]])
        target_points = np.array([[2, 2], [1, 1], [1, 2], [3, 7]])
        expected = [True, True, True, True, True, True]
        ins, hu = ConvexHullUtil.inside_hull(
            hull_points=hull_points, target_points=hull_points)
        np_test.assert_equal(ins, expected)
        expected = [True, True, True, False]
        ins, hu = ConvexHullUtil.inside_hull(
            hull_points=hull_points, target_points=target_points)
        np_test.assert_equal(ins, expected)

        # 3d defined points
        hull_points = np.array(
            [[2, 1, 1], [5, 5, 1.5], [4, 6, 2], [1, 2, 1],
             [3, 2, 5], [6, 7, 6], [5, 8, 7], [2, 3, 6],
             [2, 1.5, 3], [5, 7, 4]])
        target_points = np.array(
            [[2, 2, 2], [5.5, 6.5, 5],
             [4, 6, 2], [3, 2, 5],
             [1, 1, 1], [5, 7, 3]])
        expected = [True, True, True, True, True, True, True, True, True, True]
        ins, hu = ConvexHullUtil.inside_hull(
            hull_points=hull_points, target_points=hull_points)
        np_test.assert_equal(ins, expected)
        expected = [True, True, True, True, False, False]
        ins, hu = ConvexHullUtil.inside_hull(
            hull_points=hull_points, target_points=target_points)
        np_test.assert_equal(ins, expected)

        # 3d random points, also test expand
        rng = np.random.default_rng(1234)
        points = rng.random((50, 3))
        points[:, 0] = 2 * points[:, 0] + 1 
        points[:, 1] = 3 * points[:, 1] + 1
        points[:, 2] = 3 * points[:, 2] + 5
        targets = np.array(
            [[2, 2, 6], [2, 2.5, 6], [1.5, 3, 7],
             [0.5, 2, 6], [1.5, 5, 6], [1.2, 3.2, 2]])
        expected = [True, True, True, False, False, False]
        ins, hu = ConvexHullUtil.inside_hull(
            hull_points=points, target_points=targets)
        np_test.assert_equal(ins, expected)
        expected = [True, True, True, True, False, False]
        ins, hu = ConvexHullUtil.inside_hull(
            hull_points=points, target_points=targets, expand=1)
        np_test.assert_equal(ins, expected)
        expected = [True, True, True, True, True, False]
        ins, hu = ConvexHullUtil.inside_hull(
            hull_points=points, target_points=targets, expand=2)
        np_test.assert_equal(ins, expected)

    def test_intersect_segment(self):
        """Tests intersect_segment()
        """

        # ndarray, None segment id
        hull_points = np.array([
            [1, 1, 1], [6, 1, 1], [1, 5, 1], [1, 1, 4],
            [2, 2, 2], [4, 2, 2], [2, 3, 2], [2, 2, 3]])
        data_x1 = np.zeros((8, 8, 8), dtype=int)
        data_x1[:, :, 1] = 2
        desired = np.zeros_like(data_x1)
        desired[:, :, 1] = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]])
        desired_all_0 = np.zeros((8, 8), dtype=int) 
        actual, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=data_x1)
        np_test.assert_equal(actual, desired)

        # ndarray and Labels, None and good segment_id
        data_z3 = np.zeros((8, 8, 8), dtype=int)
        data_z3[:, :, 3] = 3
        desired_z3 = np.zeros_like(data_z3)
        desired_z3[:, :, 3] = np.array([
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 1, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]])
        actual, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=data_z3, segment_id=3)
        np_test.assert_equal(actual, desired_z3)
        actual, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=Labels(data_z3), segment_id=None)
        np_test.assert_equal(actual.data, desired_z3)
        actual, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=Labels(data_z3), segment_id=3)
        np_test.assert_equal(actual.data, desired_z3)

        # file, segment id None, good, check pixelsize
        seg = Labels(data_z3.astype('int16'))
        seg.write(file=self.intersect_path_in)
        _, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=self.intersect_path_in,
            segment_id=None, out_path=self.intersect_path_out)
        actual = Labels.read(file=self.intersect_path_out)
        np_test.assert_equal(actual.data, desired_z3)
        seg = Labels(data_z3.astype('int16'))
        seg.write(file=self.intersect_path_in, pixel=2.5)
        _, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=self.intersect_path_in,
            segment_id=3, out_path=self.intersect_path_out)
        actual = Labels.read(file=self.intersect_path_out, header=True)
        np_test.assert_equal(actual.data, desired_z3)
        np_test.assert_almost_equal(actual.pixelsize, 2.5)
              
        # ndarray and Labels, bad segment_id
        desired_all_zeros = np.zeros_like(data_z3)
        actual, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=data_z3, segment_id=2)
        np_test.assert_equal(actual, desired_all_zeros)
        actual, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=Labels(data_z3), segment_id=2)
        np_test.assert_equal(actual.data, Labels(desired_all_zeros).data)

        # expand
        data_z5 = np.zeros((8, 8, 8), dtype=int)
        data_z5[:, :, 5] = 5
        desired_z5_exp0 = np.zeros_like(data_z5)
        desired_z5_exp1 = np.zeros_like(data_z5)
        desired_z5_exp1[:, :, 5] = np.array([
            [1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0]])
        actual, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=data_z5)
        np_test.assert_equal(actual, desired_z5_exp0)
        actual, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=Labels(data_z5), expand=1)
        np_test.assert_equal(actual.data, desired_z5_exp1)

        # file, expand in pix, sufficient to get to z=5
        seg = Labels(data_z5.astype('int16'))
        seg.write(file=self.intersect_path_in, pixel=1.5)
        _, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=self.intersect_path_in,
             out_path=self.intersect_path_out, expand=1)
        actual = Labels.read(file=self.intersect_path_out, header=True)
        np_test.assert_equal(actual.data, desired_z5_exp1)

        # file expand in nm, doesn't get to z=5 
        _, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=self.intersect_path_in,
             out_path=self.intersect_path_out, expand_nm=1)
        actual = Labels.read(file=self.intersect_path_out, header=True)
        np_test.assert_equal(actual.data, desired_z5_exp0)

        # file expand more in nm, gets to z=5 
        _, hull = ConvexHullUtil.intersect_segment(
            hull_points=hull_points, segment=self.intersect_path_in,
             out_path=self.intersect_path_out, expand_nm=1.5)
        actual = Labels.read(file=self.intersect_path_out, header=True)
        np_test.assert_equal(actual.data, desired_z5_exp1)
        
    def tearDown(self):
        """Removes leftover files
        """

        try:
            os.remove(self.intersect_path_in)
        except FileNotFoundError:
            pass
        try:
            os.remove(self.intersect_path_out)
        except FileNotFoundError:
            pass

        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(
        TestConvexHullUtil)
    unittest.TextTestRunner(verbosity=2).run(suite)
