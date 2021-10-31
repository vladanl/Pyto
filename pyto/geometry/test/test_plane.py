"""

Tests module plane

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

import unittest

import numpy as np
import numpy.testing as np_test

from pyto.geometry.plane import Plane


class TestPlane(np_test.TestCase):
    """
    """

    def setUp(self):
        pass

    def test_get_normal(self):
        """
        Tests get_normal()
        """

        # 2D
        points = [[1, 2], [2, 3]]
        plane = Plane()
        normal = plane.get_normal(npoints=points)
        desired = np.array([1, -1]) * np.sqrt(2)/2
        try:
            np_test.assert_almost_equal(normal, desired)
        except AssertionError:
            np_test.assert_almost_equal(normal, -desired)

        # 2D
        points = [[1, 2], [3, 3]]
        pl = Plane()
        normal = pl.get_normal(npoints=points)
        desired = np.array([1, -2]) / np.sqrt(5)
        try:
            np_test.assert_almost_equal(normal, desired)
        except AssertionError:
            np_test.assert_almost_equal(normal, -desired)

        # 2D, along x
        points = [[2, 1], [3, 1]]
        pl = Plane()
        normal = pl.get_normal(npoints=points)
        desired = np.array([0, 1])
        try:
            np_test.assert_almost_equal(normal, desired)
        except AssertionError:
            np_test.assert_almost_equal(normal, -desired)

        # 2D, degenerate along x
        points = [[2, 0], [3, 0]]
        pl = Plane()
        normal = pl.get_normal(npoints=points)
        desired = np.array([0, 1])
        try:
            np_test.assert_almost_equal(normal, desired)
        except AssertionError:
            np_test.assert_almost_equal(normal, -desired)

        # 3D
        points = [[0, 0, 2], [2, 0, 0], [0, -2, 0]]
        pl = Plane()
        normal = pl.get_normal(npoints=points)
        desired = np.array([1, -1, 1]) * np.sqrt(3)/3
        try:
            np_test.assert_almost_equal(normal, desired)
        except AssertionError:
            np_test.assert_almost_equal(normal, -desired)

        # 3D, along xz
        points = [[1, 0, 1], [1, 0, 2], [-1, 0, 4]]
        pl = Plane()
        normal = pl.get_normal(npoints=points)
        desired = np.array([0, 1, 0])
        try:
            np_test.assert_almost_equal(np.abs(normal), desired)
        except AssertionError:
            np_test.assert_almost_equal(normal, -desired)

    def test_make(self):
        """
        Tests make()
        """

        # 2D
        normal = [0, 1]
        point = [2, 1]
        shape = (5, 5)
        pl = Plane.make(shape=shape, normal=normal, point=point).data
        desired = np.ones(shape)
        desired[:, 1] = 0
        desired[:, 0] = -1
        np_test.assert_equal(pl, desired)

        # 2D, 0-thickness
        normal = [1, 1]
        point = [2, 1]
        pl = Plane.make(shape=(5, 5), normal=normal, point=point, thick=0).data
        np_test.assert_equal(pl[2, 1], 0)
        np_test.assert_equal(pl[3, 0], 0)
        np_test.assert_equal(pl[1, 2], 0)
        np_test.assert_equal(pl[0, 3], 0)
        np_test.assert_equal(pl[2, 2], 1)
        np_test.assert_equal(pl[3, 1], 1)
        np_test.assert_equal(pl[1, 1], -1)
        np_test.assert_equal(pl[2, 0], -1)
        np_test.assert_equal((pl == 0).sum(), 4)
        np_test.assert_equal((pl == 1).sum(), 15)
        np_test.assert_equal((pl == -1).sum(), 6)

        # 2D, 0-thickness, points
        points = [[2, 1], [3, 0]]
        pl = Plane.make(shape=(5, 5), npoints=points, thick=0).data
        np_test.assert_equal(pl[2, 1], 0)
        np_test.assert_equal(pl[3, 0], 0)
        np_test.assert_equal(pl[1, 2], 0)
        np_test.assert_equal(pl[0, 3], 0)
        np_test.assert_equal(pl[2, 2], 1)
        np_test.assert_equal(pl[3, 1], 1)
        np_test.assert_equal(pl[1, 1], -1)
        np_test.assert_equal(pl[2, 0], -1)
        np_test.assert_equal((pl == 0).sum(), 4)
        np_test.assert_equal((pl == 1).sum(), 15)
        np_test.assert_equal((pl == -1).sum(), 6)

        # 2D, non-int point
        normal = [-1, 1]
        point = [2.1, 1.2]
        pl = Plane.make(shape=(5, 5), normal=normal, point=point).data
        np_test.assert_equal(pl[2, 1], 0)
        np_test.assert_equal(pl[3, 2], 0)
        np_test.assert_equal(pl[4, 3], 0)
        np_test.assert_equal(pl[1, 0], 0)
        np_test.assert_equal(pl[2, 2], 1)
        np_test.assert_equal(pl[3, 3], 1)
        np_test.assert_equal(pl[3, 1], -1)
        np_test.assert_equal(pl[2, 0], -1)
        np_test.assert_equal((pl == 0).sum(), 4)
        np_test.assert_equal((pl == 1).sum(), 15)
        np_test.assert_equal((pl == -1).sum(), 6)

        # 3D
        normal = [-1, -1, -1]
        point = [1, 2, 3]
        plane_lab, pos_lab, neg_lab = (2, 3, 4)
        pl = Plane.make(
            shape=(5, 5, 5), normal=normal, point=point).data
        np_test.assert_equal(pl[1, 2, 3], 0)
        np_test.assert_equal(pl[2, 1, 3], 0)
        np_test.assert_equal(pl[1, 1, 4], 0)
        np_test.assert_equal(pl[1, 2, 2], 1)
        np_test.assert_equal(pl[0, 2, 3], 1)
        np_test.assert_equal(pl[1, 3, 3], -1)
        np_test.assert_equal(pl[3, 2, 3], -1)

        # 3D points, same plane as previous
        points = [[1, 2, 3], [2, 1, 3], [1, 1, 4]]
        pl_previous = pl
        pl = Plane.make(shape=(5, 5, 5), npoints=points).data
        np_test.assert_equal(pl[1, 2, 3], 0)
        np_test.assert_equal(pl[2, 1, 3], 0)
        np_test.assert_equal(pl[1, 1, 4], 0)
        left_handed = []
        try:
            np_test.assert_almost_equal(pl, pl_previous)
            left_handed.append(False)
        except AssertionError:
            np_test.assert_almost_equal(pl, -pl_previous)
            left_handed.append(True)

        # 3D points, same as previous, but should have the opposite direction
        points = [[1, 2, 3], [1, 1, 4], [2, 1, 3]]
        pl = Plane.make(shape=(5, 5, 5), npoints=points).data
        np_test.assert_equal(pl[1, 2, 3], 0)
        np_test.assert_equal(pl[2, 1, 3], 0)
        np_test.assert_equal(pl[1, 1, 4], 0)
        try:
            np_test.assert_almost_equal(pl, pl_previous)
            left_handed.append(True)
        except AssertionError:
            np_test.assert_almost_equal(pl, -pl_previous)
            left_handed.append(False)

        # 3D, different labels
        normal = [-1, -1, -1]
        point = [1, 2, 3]
        plane_lab, pos_lab, neg_lab = (2, 3, 4)
        pl = Plane.make(
            shape=(5, 5, 5), normal=normal, point=point,
            plane_label=plane_lab, positive_label=pos_lab,
            negative_label=neg_lab).data
        np_test.assert_equal(pl[1, 2, 3], plane_lab)
        np_test.assert_equal(pl[2, 1, 3], plane_lab)
        np_test.assert_equal(pl[1, 1, 4], plane_lab)
        np_test.assert_equal(pl[1, 2, 2], pos_lab)
        np_test.assert_equal(pl[0, 2, 3], pos_lab)
        np_test.assert_equal(pl[1, 3, 3], neg_lab)
        np_test.assert_equal(pl[3, 2, 3], neg_lab)

        # 3D xz plane points
        points = [[3.3, 2, 2.2], [3.4, 2, 4.4], [1.1, 2, 4.5]]
        pl = Plane.make(shape=(5, 5, 5), npoints=points).data
        np_test.assert_equal(pl[1, 2, 3], 0)
        np_test.assert_equal(pl[4, 2, 4], 0)
        try:
            np_test.assert_equal(pl[4, 3, 4], 1)
            left_handed.append(True)
        except AssertionError:
            np_test.assert_equal(pl[4, 3, 4], -1)
            left_handed.append(False)

        # 3D xz plane points
        points = [[0, 2, 2.2], [0, 2, 4.4], [0, 3, 4.5]]
        pl = Plane.make(shape=(5, 5, 5), npoints=points).data
        np_test.assert_equal(pl[0, 2, 3], 0)
        np_test.assert_equal(pl[0, 2, 4], 0)
        try:
            np_test.assert_equal(pl[4, 3, 4], 1)
            left_handed.append(True)
        except AssertionError:
            np_test.assert_equal(pl[4, 3, 4], -1)
            left_handed.append(False)

        # print info about direction of normal and order of points
        if np.all(left_handed):
            outcome = 'left handed'
        elif not np.any(left_handed):
            outcome = 'right handed'
        else:
            outcome = 'mixed left and right handed'
        #print(left_handed)
        #print(
        #    ("INFO: Direction of the 2D plane normal is {} in respect to the"
        #     + " three specified points on the plane").format(outcome))

        # 3D all points on a line
        points = [[0, 2, 2], [0, 3, 3], [0, 4, 4]]
        with np_test.assert_raises(ValueError):
            Plane.make(shape=(5, 5, 5), npoints=points)

        # 4D points, x-slice
        points = [
            [1, 2, 2.2, 1], [1, 2, 4.4, 5], [1, 3, 4.5, 4], [1, 1, 2.5, 4]]
        pl = Plane.make(shape=(5, 5, 5, 5), npoints=points).data
        np_test.assert_equal(pl[1, 2, 3, 2], 0)
        np_test.assert_equal(pl[1, 2, 4, 3], 0)
        try:
            np_test.assert_equal(pl[2, 3, 4, 1], 1)
            np_test.assert_equal(pl[0, 4, 2, 2], -1)
        except AssertionError:
            np_test.assert_equal(pl[2, 3, 4, 1], -1)
            np_test.assert_equal(pl[0, 4, 2, 2], 1)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestPlane)
    unittest.TextTestRunner(verbosity=2).run(suite)
