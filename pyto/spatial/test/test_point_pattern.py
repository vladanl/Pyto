"""
Tests module point_function

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision:"

import os
import unittest
from functools import partial

import numpy as np
import numpy.testing as np_test
from numpy.random import default_rng
import scipy as sp
from scipy.spatial.distance import cdist, pdist, squareform
import pandas as pd

from pyto.spatial.test import common
import pyto.spatial.point_pattern as pattern

class TestPointPattern(np_test.TestCase):
    """
    Tests point_pattern module
    """

    def setUp(self):
        """
        """
        pass

    def test_random_rectangle(self):
        """Tests random_restangle, random_rectangle_fun and pattern_exclusion()
        """

        excl = 1.1
        # 42 places for 10 points - just about always possible
        actual = pattern.random_rectangle(
            N=10, rectangle=[[1, 3], [8, 9]], exclusion=excl, mode='rough',
            max_iter=20)
        np_test.assert_equal((pdist(actual) < excl).any(), False)
        
        actual = pattern.random_rectangle(
            N=10, rectangle=[[1, 3], [8, 9]], exclusion=1.1, mode='fine',
            max_iter=20)
        np_test.assert_equal((pdist(actual) < excl).any(), False)

        # 18 places for 10 points - no way
        with np_test.assert_raises(ValueError):
            pattern.random_rectangle(
                N=10, rectangle=[[1, 3], [7, 6]], exclusion=1.1, mode='fine',
                max_iter=20)
         
    def test_random_region(self):
        """Tests random_region, random_region_fun and pattern_exclusion()
        """

        excl = 1.1
        
        # image rectangle, 42 places for 10 points - just about always possible
        rectangle = np.array([[1, 3], [8, 9]])
        image = np.zeros((15, 15), dtype=int)
        image[slice(*rectangle[:, 0]), slice(*rectangle[:, 1])] = 2
        actual = pattern.random_region(
            N=10, region=image, region_id=2, exclusion=excl, mode='rough',
            max_iter=20)
        np_test.assert_equal((pdist(actual) < excl).any(), False)
        np_test.assert_equal(
            ((actual[:, 0] >= rectangle[0, 0]) & (actual[:, 0] < rectangle[1, 0])
             & (actual[:, 1] >= rectangle[0, 1])
             & (actual[:, 1] < rectangle[1, 1])).all(), True)

        # points rectangle, 42 places for 10 points - just about always possible
        rectangle = np.array([[1, 3], [8, 9]])
        size = rectangle[1, :] - rectangle[0, :]
        offset = rectangle[0]
        coords = np.swapaxes(np.indices(size), 0, 2).reshape(-1, 2) + offset
        actual = pattern.random_region(
            N=10, region_coords=coords, exclusion=excl, mode='fine',
            max_iter=20)
        np_test.assert_equal((pdist(actual) < excl).any(), False)
        np_test.assert_equal(
            ((actual[:, 0] >= rectangle[0, 0]) & (actual[:, 0] < rectangle[1, 0])
             & (actual[:, 1] >= rectangle[0, 1])
             & (actual[:, 1] < rectangle[1, 1])).all(), True)

        # 18 places for 10 points - no way
        rectangle = np.array([[1, 3], [7, 6]])
        image = np.zeros((15, 15), dtype=int)
        image[slice(*rectangle[:, 0]), slice(*rectangle[:, 1])] = 2
        with np_test.assert_raises(ValueError):
            pattern.random_region(
                N=10, region=image, region_id=2, exclusion=excl, mode='fine',
                max_iter=200)
       
        # 0 points
        rectangle = np.array([[1, 3], [7, 6]])
        image = np.zeros((15, 15), dtype=int)
        image[slice(*rectangle[:, 0]), slice(*rectangle[:, 1])] = 2
        actual = pattern.random_region(
            N=0, region=image, region_id=2, exclusion=excl, mode='fine',
            max_iter=200)
        np_test.assert_array_equal(actual, np.array([]))

    def test_pattern_exclusion(self):
        """Tests pattern_exclusion()
        """

        rng = default_rng()

        # points that require no exclusion
        points = np.array([[1, 2], [4, 4], [7, 8]])
        def pattern_fun(N, rng, points):
            return points
        pattern_fun_actual = partial(pattern_fun, points=points)
        actual = pattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=points.shape[0], exclusion=2)
        np_test.assert_array_equal(
            np.sort(actual, axis=0), np.sort(points, axis=0))

        # exclusion used
        actual = pattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=2, exclusion=4)
        np_test.assert_array_equal(actual.shape[0], 2)
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(points == act, axis=1).any()
                for act in actual]),
            2)

        # uses random_regions_fun(), w/wo other
        points = np.array([[2, 2], [2, 3], [2, 4], [3, 4]])
        pattern_fun_actual = partial(pattern.random_region_fun, coords=points)
        actual = pattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=2, exclusion=1.1)
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(points == act, axis=1).any()
                for act in actual]),
            2)
        actual = pattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=1, exclusion=1.1,
            other=[[2, 3]])
        np_test.assert_array_equal(actual, [[3, 4]])
        actual = pattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=1, exclusion=1.1,
            other=[[2, 4], [3, 4]])
        np_test.assert_array_equal(actual, [[2, 2]])
           
        # uses random_rectangle_fun(), w/wo other
        rectangle = np.array([[2, 2], [5, 6]])
        points = np.array(
            [[x, y] for x in range(rectangle[0, 0], rectangle[1, 0])
             for y in range(rectangle[0, 1], rectangle[1, 1])])
        pattern_fun_actual = partial(
            pattern.random_rectangle_fun, rectangle=rectangle)
        actual = pattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=12, exclusion=0.8)
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(points == act, axis=1).any()
                for act in actual]),
            12)
        other = np.array([[2, 4], [3, 4], [4, 4]])
        actual = pattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=1, exclusion=1.2, other=other)
        desired = np.array([[2, 2], [3, 2], [4, 2]])
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(points == act, axis=1).any()
                for act in actual]),
            1)
        with np_test.assert_raises(ValueError):
            pattern.pattern_exclusion(
                pattern_fun=pattern_fun_actual, N=3, exclusion=1.2, other=other)
            
    def test_exclude(self):
        """Tests exclude()
        """

        # no points
        np_test.assert_equal(
            pattern.exclude(points=None, exclusion=1), np.array([]))
        
        # no points
        np_test.assert_equal(
            pattern.exclude(points=[], exclusion=1), np.array([]))
       
        # one point
        np_test.assert_equal(
            pattern.exclude(points=np.array([[1, 2]]), exclusion=1.1),
            np.array([[1, 2]]))
        
        p1 = np.array(
            [[2, 2], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
             [7, 8], [6, 9], [6, 6], [2, 1], [7, 7], [7, 7]])

        # rough
        desired = np.array([[2, 2], [4, 4], [7, 8], [6, 9], [6, 6]])
        actual = pattern.exclude(points=p1, exclusion=1.1, mode='rough')
        np_test.assert_equal(actual, desired)

        # fine
        desired = np.array(
            [[2, 2], [4, 4], [7, 8], [6, 9], [6, 6], [4, 6], [4, 8]])
        actual = pattern.exclude(points=p1, exclusion=1.1, mode='fine')
        np_test.assert_equal(actual, desired)
        
        # fine, different exclusion
        desired = np.array([[2, 2], [4, 4], [7, 8], [6, 6], [4, 6], [4, 8]])
        actual = pattern.exclude(points=p1, exclusion=1.5, mode='fine')
        np_test.assert_equal(actual, desired)

        # other
        p2 = np.array([[4, 6], [7, 9]])
        desired = np.array(
            [[2, 2], [4, 4], [4, 8], [6, 6], [2, 1], [7, 7], [7, 7]])
        actual = pattern.exclude(points=p1, other=p2, exclusion=1.5)
        np_test.assert_equal(actual, desired)

    def test_cocluster_region(self):
        """Test cocluster_region()
        """

        # p = 0 or 1
        region = np.zeros((20, 20), dtype=int)
        region[5:15, 10:20] = 2
        clusters = [[5, 13], [12, 14]]
        desired_hood = [
            [5, 13], [5, 12], [5, 14], [6, 13],
            [12, 14], [11, 14], [13, 14], [12, 13], [12, 15]]
        actual = pattern.cocluster_region(
            clusters=clusters, N=20, p_cluster=1, region=region, region_id=2,
            max_dist=1)
        np_test.assert_array_equal(actual.shape, [20, 2])
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(desired_hood == act, axis=1).any()
                for act in actual]),
            20)
        actual = pattern.cocluster_region(
            clusters=clusters, N=20, p_cluster=0, region=region, region_id=2,
            max_dist=1)
        np_test.assert_array_equal(actual.shape, [20, 2])

        # other p, seed fixed
        actual = pattern.cocluster_region(
            clusters=clusters, N=10, p_cluster=0.6, region=region, region_id=2,
            max_dist=1, seed=222)
        desired_222 = np.array(
            [[5, 12], [ 5, 12], [ 5, 12], [ 5, 13], [ 6, 13],
             [12, 15], [11, 14], [11, 14], [13, 14], [12, 14]])
        np_test.assert_array_equal(actual.shape, [10, 2])
        np_test.assert_array_equal(actual.dtype == int, True)
        np_test.assert_array_equal(actual, desired_222)

        # other p, with exclusion
        region = np.zeros((20, 20), dtype=int)
        region[5:15, 10:20] = 2
        clusters = [[5, 13], [12, 14]]
        desired_hood = [
            [5, 13], [5, 12], [5, 14], [6, 13],
            [12, 14], [11, 14], [13, 14], [12, 13], [12, 15]]
        actual = pattern.cocluster_region(
            clusters=clusters, N=4, p_cluster=1, region=region, region_id=2,
            max_dist=1, exclusion=0.6)
        np_test.assert_array_equal(actual.shape, [4, 2])
        np_test.assert_array_equal(
            [np.logical_and.reduce(desired_hood == act, axis=1).sum()
             for act in actual],
            [1, 1, 1, 1])
        
    def test_random_hoods(self):
        """Test random_hoods()
        """

        # one cluster
        region_coords = np.indices((4, 5)).reshape(2, -1).transpose()
        desired = np.array([[2, 3], [2, 2], [2, 4], [1, 3], [3, 3]]) 
        actual = pattern.random_hoods(
            clusters=[[2, 3]], n_points=[10], region_coords=region_coords,
            max_dist=1)
        np_test.assert_array_equal(actual.shape, [10, 2])
        np_test.assert_equal(np.all([act in desired for act in actual]), True)

        # cluster hood outside region
        region_coords = np.indices((4, 5)).reshape(2, -1).transpose()
        desired_0 = np.array([[0, 3], [0, 2], [0, 4], [1, 3]])
        desired_1 = np.array([[7, 4], [7, 3], [7, 5], [6, 4], [8, 4]])
        with np_test.assert_raises(ValueError):
            actual = pattern.random_hoods(
                clusters=[[0, 3], [7, 4]], n_points=[6, 4],
                region_coords=region_coords, max_dist=1)
        
        # >1 cluster
        region_coords = np.indices((10, 10)).reshape(2, -1).transpose()
        desired_0 = np.array([[0, 3], [0, 2], [0, 4], [1, 3]])
        desired_1 = np.array([[7, 4], [7, 3], [7, 5], [6, 4], [8, 4]])
        actual = pattern.random_hoods(
            clusters=[[0, 3], [7, 4]], n_points=[6, 4],
            region_coords=region_coords, max_dist=1)
        np_test.assert_array_equal(actual.shape, [10, 2])
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(desired_0 == act, axis=1).any()
                for act in actual]),
            6)
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(desired_1 == act, axis=1).any()
                for act in actual]),
            4)
        
        # >1 cluster, one cluster 0
        region_coords = np.indices((10, 10)).reshape(2, -1).transpose()
        desired_0 = np.array([[0, 3], [0, 2], [0, 4], [1, 3]])
        desired_1 = np.array([[7, 4], [7, 3], [7, 5], [6, 4], [8, 4]])
        actual = pattern.random_hoods(
            clusters=[[0, 3], [7, 4]], n_points=[0, 4],
            region_coords=region_coords, max_dist=1)
        np_test.assert_array_equal(actual.shape, [4, 2])
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(desired_0 == act, axis=1).any()
                for act in actual]),
            0)
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(desired_1 == act, axis=1).any()
                for act in actual]),
            4)

        # with exclusion = 0.5
        region_coords = np.indices((10, 10)).reshape(2, -1).transpose()
        desired_0 = np.array([[0, 3], [0, 2], [0, 4], [1, 3]])
        desired_1 = np.array([[7, 4], [7, 3], [7, 5], [6, 4], [8, 4]])
        actual = pattern.random_hoods(
            clusters=[[0, 3], [7, 4]], n_points=[4, 5],
            region_coords=region_coords, max_dist=1, exclusion=0.5)
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(desired_0 == act, axis=1).any()
                for act in actual]),
            4)
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(desired_1 == act, axis=1).any()
                for act in actual]),
            5)
        np_test.assert_equal(
            np.sort(actual, axis=0),
            np.sort(np.vstack([desired_0, desired_1]), axis=0))

        # with exclusion > max_dist
        region_coords = np.indices((10, 10)).reshape(2, -1).transpose()
        desired_0 = np.array([[0, 3], [0, 2], [0, 4], [1, 3]])
        desired_1 = np.array([[7, 4], [7, 3], [7, 5], [6, 4], [8, 4]])
        actual = pattern.random_hoods(
            clusters=[[0, 3], [7, 4]], n_points=[1, 1],
            region_coords=region_coords, max_dist=1, exclusion=2)
        np_test.assert_array_equal(actual.shape, [2, 2])
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(desired_0 == act, axis=1).any()
                for act in actual]),
            1)
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(desired_1 == act, axis=1).any()
                for act in actual]),
            1)
         
    def test_get_n_points_cluster(self):
        """Tests get_n_points_cluster()
        """

        actual = pattern.get_n_points_cluster(
            n_clusters=1, n_points=10, p_cluster=1)
        np_test.assert_equal(actual, [10])
        actual = pattern.get_n_points_cluster(
            n_clusters=3, n_points=10, p_cluster=0)
        np_test.assert_array_equal(actual, [0, 0, 0])
        actual = pattern.get_n_points_cluster(
            n_clusters=3, n_points=10, p_cluster=1)
        np_test.assert_array_equal(np.sum(actual), 10)
        actual = pattern.get_n_points_cluster(
            n_clusters=3, n_points=10, p_cluster=0.6, seed=123)
        np_test.assert_array_equal(actual, [2, 2, 0])
        actual = pattern.get_n_points_cluster(
            n_clusters=4, n_points=20, p_cluster=0.6, seed=1234)
        np_test.assert_array_equal(actual, [1, 2, 1, 5])
      
    def test_get_region_coords(self):
        """Tests get_region_coords()
        """

        image = np.zeros((10, 10), dtype=int)
        image[1:4, 2] = 2
        image[6, 2:4] = 5
        expected_2 = np.array([[1, 2], [2, 2], [3, 2]])
        expected_5 = np.array([[6, 2], [6, 3]])
        actual = pattern.get_region_coords(region=image, region_id=2)
        np_test.assert_array_equal(np.sort(actual, axis=0), expected_2)
        actual = pattern.get_region_coords(region=image, region_id=5)
        np_test.assert_array_equal(np.sort(actual, axis=0), expected_5)
        actual = pattern.get_region_coords(
            region=image, region_id=2, shuffle=True)
        np_test.assert_array_equal(np.sort(actual, axis=0), expected_2)
        actual = pattern.get_region_coords(
            region_coords=[[1, 2], [2, 2], [3, 2]], shuffle=True)
        np_test.assert_array_equal(np.sort(actual, axis=0), expected_2)
        np_test.assert_equal(isinstance(actual, np.ndarray), True)       
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPointPattern)
    unittest.TextTestRunner(verbosity=2).run(suite)
