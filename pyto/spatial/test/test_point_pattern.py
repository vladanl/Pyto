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
import pyto.spatial.point_pattern as ppattern

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
        actual = ppattern.random_rectangle(
            N=10, rectangle=[[1, 3], [8, 9]], exclusion=excl, mode='rough',
            max_iter=20)
        np_test.assert_equal((pdist(actual) < excl).any(), False)
        
        actual = ppattern.random_rectangle(
            N=10, rectangle=[[1, 3], [8, 9]], exclusion=1.1, mode='fine',
            max_iter=20)
        np_test.assert_equal((pdist(actual) < excl).any(), False)

        # 18 places for 10 points - no way
        with np_test.assert_raises(ValueError):
            ppattern.random_rectangle(
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
        actual = ppattern.random_region(
            N=10, region=image, region_id=2, exclusion=excl, mode='rough',
            max_iter=20)
        np_test.assert_equal((pdist(actual) < excl).any(), False)
        np_test.assert_equal(
            ((actual[:, 0] >= rectangle[0, 0])
             & (actual[:, 0] < rectangle[1, 0])
             & (actual[:, 1] >= rectangle[0, 1])
             & (actual[:, 1] < rectangle[1, 1])).all(), True)

        # points rectangle, 42 places for 10 points - just about always possible
        rectangle = np.array([[1, 3], [8, 9]])
        size = rectangle[1, :] - rectangle[0, :]
        offset = rectangle[0]
        coords = np.swapaxes(np.indices(size), 0, 2).reshape(-1, 2) + offset
        actual = ppattern.random_region(
            N=10, region_coords=coords, exclusion=excl, mode='fine',
            max_iter=20)
        np_test.assert_equal((pdist(actual) < excl).any(), False)
        np_test.assert_equal(
            ((actual[:, 0] >= rectangle[0, 0])
             & (actual[:, 0] < rectangle[1, 0])
             & (actual[:, 1] >= rectangle[0, 1])
             & (actual[:, 1] < rectangle[1, 1])).all(), True)

        # 18 places for 10 points - no way
        rectangle = np.array([[1, 3], [7, 6]])
        image = np.zeros((15, 15), dtype=int)
        image[slice(*rectangle[:, 0]), slice(*rectangle[:, 1])] = 2
        with np_test.assert_raises(ValueError):
            ppattern.random_region(
                N=10, region=image, region_id=2, exclusion=excl, mode='fine',
                max_iter=200)
       
        # 0 points
        rectangle = np.array([[1, 3], [7, 6]])
        image = np.zeros((15, 15), dtype=int)
        image[slice(*rectangle[:, 0]), slice(*rectangle[:, 1])] = 2
        actual = ppattern.random_region(
            N=0, region=image, region_id=2, exclusion=excl, mode='fine',
            max_iter=200)
        np_test.assert_array_equal(actual, np.array([]))

        # test shuffle and seed, essentially no exclusion (<1)
        image = np.zeros((10, 10), dtype=int)
        image[1:4, 2] = 2
        expected_2 = np.array([[1, 2], [2, 2], [3, 2]])
        expected_2shse = np.array([[1, 2], [3, 2], [2, 2]])
        expected_2se = np.array([[2, 2], [3, 2], [1, 2]])
        actual = ppattern.random_region(
            N=3, region=image, region_id=2, exclusion=0.5, shuffle=False)
        np_test.assert_array_equal(np.sort(actual, axis=0), expected_2)
        actual = ppattern.random_region(
             N=3, region=image, region_id=2, exclusion=0.5, shuffle=True,
             seed=125)
        np_test.assert_array_equal(actual, expected_2shse)
        actual = ppattern.random_region(
             N=3, region=image, region_id=2, exclusion=0.5, shuffle=False,
             seed=125)
        np_test.assert_array_equal(actual, expected_2se)
         
    def test_pattern_exclusion(self):
        """Tests pattern_exclusion()
        """

        rng = default_rng()

        # points that require no exclusion
        points = np.array([[1, 2], [4, 4], [7, 8]])
        def pattern_fun(N, rng, points):
            return points
        pattern_fun_actual = partial(pattern_fun, points=points)
        actual = ppattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=points.shape[0], exclusion=2)
        np_test.assert_array_equal(
            np.sort(actual, axis=0), np.sort(points, axis=0))

        # exclusion used
        actual = ppattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=2, exclusion=4)
        np_test.assert_array_equal(actual.shape[0], 2)
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(points == act, axis=1).any()
                for act in actual]),
            2)

        # uses random_regions_fun(), w/wo other
        points = np.array([[2, 2], [2, 3], [2, 4], [3, 4]])
        pattern_fun_actual = partial(ppattern.random_region_fun, coords=points)
        actual = ppattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=2, exclusion=1.1)
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(points == act, axis=1).any()
                for act in actual]),
            2)
        actual = ppattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=1, exclusion=1.1,
            other=[[2, 3]])
        np_test.assert_array_equal(actual, [[3, 4]])
        actual = ppattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=1, exclusion=1.1,
            other=[[2, 4], [3, 4]])
        np_test.assert_array_equal(actual, [[2, 2]])
           
        # uses random_rectangle_fun(), w/wo other
        rectangle = np.array([[2, 2], [5, 6]])
        points = np.array(
            [[x, y] for x in range(rectangle[0, 0], rectangle[1, 0])
             for y in range(rectangle[0, 1], rectangle[1, 1])])
        pattern_fun_actual = partial(
            ppattern.random_rectangle_fun, rectangle=rectangle)
        actual = ppattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=12, exclusion=0.8)
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(points == act, axis=1).any()
                for act in actual]),
            12)
        other = np.array([[2, 4], [3, 4], [4, 4]])
        actual = ppattern.pattern_exclusion(
            pattern_fun=pattern_fun_actual, N=1, exclusion=1.2, other=other)
        desired = np.array([[2, 2], [3, 2], [4, 2]])
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(points == act, axis=1).any()
                for act in actual]),
            1)
        with np_test.assert_raises(ValueError):
            ppattern.pattern_exclusion(
                pattern_fun=pattern_fun_actual, N=3, exclusion=1.2, other=other)
            
    def test_exclude(self):
        """Tests exclude()
        """

        # no points
        np_test.assert_equal(
            ppattern.exclude(points=None, exclusion=1), np.array([]))
        
        # no points
        np_test.assert_equal(
            ppattern.exclude(points=[], exclusion=1), np.array([]))
       
        # one point
        np_test.assert_equal(
            ppattern.exclude(points=np.array([[1, 2]]), exclusion=1.1),
            np.array([[1, 2]]))
        
        p1 = np.array(
            [[2, 2], [4, 4], [4, 5], [4, 6], [4, 7], [4, 8],
             [7, 8], [6, 9], [6, 6], [2, 1], [7, 7], [7, 7]])

        # rough
        desired = np.array([[2, 2], [4, 4], [7, 8], [6, 9], [6, 6]])
        actual = ppattern.exclude(points=p1, exclusion=1.1, mode='rough')
        np_test.assert_equal(actual, desired)

        # fine
        desired = np.array(
            [[2, 2], [4, 4], [7, 8], [6, 9], [6, 6], [4, 6], [4, 8]])
        actual = ppattern.exclude(points=p1, exclusion=1.1, mode='fine')
        np_test.assert_equal(actual, desired)
        
        # fine, different exclusion
        desired = np.array([[2, 2], [4, 4], [7, 8], [6, 6], [4, 6], [4, 8]])
        actual = ppattern.exclude(points=p1, exclusion=1.5, mode='fine')
        np_test.assert_equal(actual, desired)

        # other
        p2 = np.array([[4, 6], [7, 9]])
        desired = np.array(
            [[2, 2], [4, 4], [4, 8], [6, 6], [2, 1], [7, 7], [7, 7]])
        actual = ppattern.exclude(points=p1, other=p2, exclusion=1.5)
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
        actual = ppattern.cocluster_region(
            clusters=clusters, N=20, p_cluster=1, region=region, region_id=2,
            max_dist=1)
        np_test.assert_array_equal(actual.shape, [20, 2])
        np_test.assert_equal(
            np.sum([
                np.logical_and.reduce(desired_hood == act, axis=1).any()
                for act in actual]),
            20)
        actual = ppattern.cocluster_region(
            clusters=clusters, N=20, p_cluster=0, region=region, region_id=2,
            max_dist=1)
        np_test.assert_array_equal(actual.shape, [20, 2])

        # other p, seed fixed
        actual = ppattern.cocluster_region(
            clusters=clusters, N=10, p_cluster=0.6, region=region, region_id=2,
            max_dist=1, seed=222)
        desired_222 = np.array(
            [[5, 12], [5, 12], [5, 12], [5, 13], [ 6, 13],
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
        actual = ppattern.cocluster_region(
            clusters=clusters, N=4, p_cluster=1, region=region, region_id=2,
            max_dist=1, exclusion=0.6)
        np_test.assert_array_equal(actual.shape, [4, 2])
        np_test.assert_array_equal(
            [np.logical_and.reduce(desired_hood == act, axis=1).sum()
             for act in actual],
            [1, 1, 1, 1])

        # mode 1o1
        region = np.zeros((20, 20), dtype=int)
        region[5:15, 10:20] = 2
        clusters = [[5, 13], [12, 14], [7, 17], [8, 18]]
        actual = ppattern.cocluster_region(
            clusters=clusters, N=4, mode='max1', p_cluster=1,
            region=region, region_id=2, max_dist=0)
        np_test.assert_array_equal(
            set(tuple(x) for x in actual), set(tuple(x) for x in clusters))
        
        # mode 1o1
        region = np.zeros((20, 20), dtype=int)
        region[5:15, 10:20] = 2
        clusters = [[5, 13], [12, 14], [7, 17], [8, 18]]
        desired = [[5, 13], [8, 18]]
        actual = ppattern.cocluster_region(
            clusters=clusters, N=2, mode='max1', p_cluster=1,
            region=region, region_id=2, max_dist=0, seed=123)
        np_test.assert_array_equal(actual, desired)
        
        # mode 1o1
        region = np.zeros((20, 20), dtype=int)
        region[5:15, 10:20] = 2
        clusters = [[5, 13], [12, 14], [7, 17], [8, 18]]
        desired = [[12, 14], [8, 18]]
        actual = ppattern.cocluster_region(
            clusters=clusters, N=6, mode='max1', p_cluster=0.33,
            region=region, region_id=2, max_dist=0, seed=124)
        np_test.assert_array_equal(actual[:2, :], desired)
        
    def test_random_hoods(self):
        """Test random_hoods()
        """

        # one cluster
        region_coords = np.indices((4, 5)).reshape(2, -1).transpose()
        desired = np.array([[2, 3], [2, 2], [2, 4], [1, 3], [3, 3]]) 
        actual = ppattern.random_hoods(
            clusters=[[2, 3]], n_points=[10], region_coords=region_coords,
            max_dist=1)
        np_test.assert_array_equal(actual.shape, [10, 2])
        np_test.assert_equal(np.all([act in desired for act in actual]), True)

        # cluster hood outside region
        region_coords = np.indices((4, 5)).reshape(2, -1).transpose()
        desired_0 = np.array([[0, 3], [0, 2], [0, 4], [1, 3]])
        desired_1 = np.array([[7, 4], [7, 3], [7, 5], [6, 4], [8, 4]])
        with np_test.assert_raises(ValueError):
            actual = ppattern.random_hoods(
                clusters=[[0, 3], [7, 4]], n_points=[6, 4],
                region_coords=region_coords, max_dist=1)
        
        # >1 cluster
        region_coords = np.indices((10, 10)).reshape(2, -1).transpose()
        desired_0 = np.array([[0, 3], [0, 2], [0, 4], [1, 3]])
        desired_1 = np.array([[7, 4], [7, 3], [7, 5], [6, 4], [8, 4]])
        actual = ppattern.random_hoods(
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
        
        # >1 cluster, shuffle, seed
        region_coords = np.indices((10, 10)).reshape(2, -1).transpose()
        desired_0 = np.array([[0, 2], [1, 3], [1, 3], [0, 2], [0, 3], [0, 2]])
        desired_1 = np.array([[8, 4], [7, 5], [7, 4], [8, 4]])
        actual = ppattern.random_hoods(
            clusters=[[0, 3], [7, 4]], n_points=[6, 4],
            region_coords=region_coords, max_dist=1, shuffle=True, seed=123)
        np_test.assert_equal(actual, np.concatenate((desired_0, desired_1)))
        desired_0 = np.array([[0, 3], [0, 4], [1, 3], [0, 3], [0, 3], [0, 3]])
        desired_1 = np.array([[7, 3], [7, 4], [8, 4], [7, 3]])
        actual = ppattern.random_hoods(
            clusters=[[0, 3], [7, 4]], n_points=[6, 4],
            region_coords=region_coords, max_dist=1, shuffle=True, seed=125)
        np_test.assert_equal(actual, np.concatenate((desired_0, desired_1)))
        
        # >1 cluster, one cluster 0
        region_coords = np.indices((10, 10)).reshape(2, -1).transpose()
        desired_0 = np.array([[0, 3], [0, 2], [0, 4], [1, 3]])
        desired_1 = np.array([[7, 4], [7, 3], [7, 5], [6, 4], [8, 4]])
        actual = ppattern.random_hoods(
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
        actual = ppattern.random_hoods(
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
        actual = ppattern.random_hoods(
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

        actual = ppattern.get_n_points_cluster(
            n_clusters=1, n_points=10, p_cluster=1)
        np_test.assert_equal(actual, [10])
        actual = ppattern.get_n_points_cluster(
            n_clusters=3, n_points=10, p_cluster=0)
        np_test.assert_array_equal(actual, [0, 0, 0])
        actual = ppattern.get_n_points_cluster(
            n_clusters=3, n_points=10, p_cluster=1)
        np_test.assert_array_equal(np.sum(actual), 10)
        actual = ppattern.get_n_points_cluster(
            n_clusters=3, n_points=10, p_cluster=0.6, seed=123)
        np_test.assert_array_equal(actual, [2, 2, 0])
        actual = ppattern.get_n_points_cluster(
            n_clusters=4, n_points=20, p_cluster=0.6, seed=1234)
        np_test.assert_array_equal(actual, [1, 2, 1, 5])
      
    def test_get_region_coords(self):
        """Tests get_region_coords()
        """

        image = np.zeros((10, 10), dtype=int)
        image[1:4, 2] = 2
        image[6, 2:4] = 5

        # no shuffle
        expected_2 = np.array([[1, 2], [2, 2], [3, 2]])
        expected_5 = np.array([[6, 2], [6, 3]])
        actual = ppattern.get_region_coords(
            region=image, region_id=2, shuffle=False)
        np_test.assert_array_equal(actual, expected_2)
        actual = ppattern.get_region_coords(
            region=image, region_id=5, shuffle=False)
        np_test.assert_array_equal(actual, expected_5)

        # shuffle w seed 
        expected_2sh = np.array([[1, 2], [3, 2], [2, 2]])
        actual = ppattern.get_region_coords(
            region=image, region_id=2, shuffle=True, seed=123)
        np_test.assert_array_equal(actual, expected_2sh)

        # region_coords arg specified
        actual = ppattern.get_region_coords(
            region_coords=[[1, 2], [2, 2], [3, 2]], shuffle=True)
        np_test.assert_array_equal(np.sort(actual, axis=0), expected_2)
        np_test.assert_equal(isinstance(actual, np.ndarray), True)
        region_coords=[[1, 2], [2, 2], [3, 2]]
        actual = ppattern.get_region_coords(
            region_coords=region_coords, shuffle=False)
        np_test.assert_array_equal(region_coords, region_coords)

    def test_select_by_region(self):
        """Tests select_by_region.
        """

        points = np.array([[1, 2], [2, 5], [3, 4], [5, 4]])
        region = np.ones((5, 5), dtype=int)
        actual = ppattern.select_by_region(
            pattern=points, region=region, region_id=1, shuffle=False)
        np_test.assert_equal(actual, np.array([[1, 2], [3, 4]]))
        actual = ppattern.select_by_region(
            pattern=points, region=region, region_id=1,
            fraction=0.5, shuffle=True, seed=123)
        np_test.assert_equal(actual, np.array([[1, 2], [3, 4]]))
        actual = ppattern.select_by_region(
            pattern=points, region=region, region_id=1,
            fraction=0.5, shuffle=True, seed=124)
        np_test.assert_equal(actual, np.array([[3, 4]]))

        # only fraction selection
        region = np.ones((8, 8), dtype=int)
        actual = ppattern.select_by_region(
            pattern=points, region=region, region_id=1, fraction=0.5,
            shuffle=False)
        np_test.assert_equal(actual, np.array([[1, 2], [2, 5]]))
         
       # both selections
        region = np.ones((6, 5), dtype=int)
        actual = ppattern.select_by_region(
            pattern=points, region=region, region_id=1, fraction=0.75,
            shuffle=False)
        np_test.assert_equal(actual, np.array([[1, 2], [3, 4]]))
         
    def test_get_rectangle_coords(self):
        """Tests get_rectangle_coords()
        """

        rec = [[10, 15, 30], [14, 17, 31]]
        desired = np.array(
            [[10, 15, 30],
             [10, 16, 30],
             [11, 15, 30],
             [11, 16, 30],
             [12, 15, 30],
             [12, 16, 30],
             [13, 15, 30],
             [13, 16, 30]])
        actual = ppattern.get_rectangle_coords(rec)
        np_test.assert_equal(actual, desired)


    def test_colocalize_pattern(self):
        """Tests colocalize_pattern()
        """

        fixed = np.array(
            [[14, 36, 67],
             [32, 12, 50],
             [20, 28, 48],
             [25, 40, 45]])
        
        region_coords = ppattern.get_rectangle_coords(
            [[10, 10, 40], [30, 50, 70]])
        actual = ppattern.colocalize_pattern(
            fixed_pattern=fixed, n_colocalize=5, region_coords=region_coords)
        np_test.assert_equal(
            np.asarray([np.asarray([(act == po).all() for po in fixed]).any()
                        for act in actual]).all(),
            True)
        desired = np.array(
            [[25, 40, 45], [25, 40, 45], [14, 36, 67],
             [22, 42, 45], [17, 44, 67], [10, 27, 48]])
        actual = ppattern.colocalize_pattern(
            fixed_pattern=fixed, n_colocalize=6, region_coords=region_coords,
            fixed_fraction=0.8, colocalize_fraction=0.4, seed=125)
        np_test.assert_equal(actual, desired)

        # shuffle_fixed False
        desired = np.tile(fixed[0, :], (6, 1))
        actual = ppattern.colocalize_pattern(
            fixed_pattern=fixed, n_colocalize=6, region_coords=region_coords,
            fixed_fraction=0.25, colocalize_fraction=1, shuffle_fixed=False,
            shuffle_region=True)
        np_test.assert_equal(actual, desired)
        desired = np.tile(fixed[1, :], (6, 1))
        
        region_coords = ppattern.get_rectangle_coords(
            [[10, 10, 40], [50, 50, 70]])
        actual = ppattern.colocalize_pattern(
            fixed_pattern=fixed[1:, :], n_colocalize=6,
            region_coords=region_coords,
            fixed_fraction=0.33, colocalize_fraction=1, shuffle_fixed=False,
            shuffle_region=True)
        np_test.assert_equal(actual, desired)

        # mode 'max1'
        region_coords = ppattern.get_rectangle_coords(
            [[10, 10, 40], [50, 50, 70]])
        actual = ppattern.colocalize_pattern(
            fixed_pattern=fixed, n_colocalize=6, mode='max1',
            region_coords=region_coords,
            fixed_fraction=1, colocalize_fraction=0.66)
        np_test.assert_equal(
            np.asarray([np.asarray([(act == po).all() for po in fixed]).any()
                        for act in actual[:3, :]]).all(),
            True)
        
        # mode 'kd'
        region_coords = ppattern.get_rectangle_coords(
            [[10, 10, 40], [50, 50, 70]])
        actual = ppattern.colocalize_pattern(
            fixed_pattern=fixed, n_colocalize=6, mode='kd',
            region_coords=region_coords, fixed_fraction=1)
        np_test.assert_equal(
            np.asarray([np.asarray([(act == po).all() for po in fixed]).any()
                        for act in actual[:4, :]]).all(),
            True)
        
        # mode 'kd'
        region_coords = ppattern.get_rectangle_coords(
            [[10, 10, 40], [50, 50, 70]])
        actual = ppattern.colocalize_pattern(
            fixed_pattern=fixed, n_colocalize=6, mode='kd',
            region_coords=region_coords, fixed_fraction=0.75)
        np_test.assert_equal(
            np.asarray([np.asarray([(act == po).all() for po in fixed]).any()
                        for act in actual[:3, :]]).all(),
            True)
        
       
    
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPointPattern)
    unittest.TextTestRunner(verbosity=2).run(suite)
