"""

Tests module parallelogram

# Author: Vladan Lucic
# $Id$
"""

__version__ = "$Revision$"

import unittest

import numpy as np
import numpy.testing as np_test

from pyto.geometry.affine_2d import Affine2D
from pyto.geometry.affine_3d import Affine3D
from pyto.geometry.parallelogram import Parallelogram


class TestParallelogram(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """
        # as set in np_test
        decimal = 7
        self.precision = 1.5 * 10**(-decimal)

    def test_make(self):
        """
        Tests make()
        """

        # 2D simple
        corners = [[1, 2], [1, 4], [3, 2]]
        rec = Parallelogram.make(shape=(5, 6), corners=corners)
        np_test.assert_equal((rec.data == 0).sum(), 8)
        np_test.assert_equal((rec.data == -1).sum(), 1)
        np_test.assert_equal((rec.data == 1).sum(), 21)
        np_test.assert_equal(rec.data[3, 4], 0)
        np_test.assert_equal(rec.data[2, 4], 0)
        np_test.assert_equal(rec.data[3, 3], 0)
        np_test.assert_equal(rec.data[2, 3], -1)
        np_test.assert_equal(rec.data[4, 4], 1)
        np_test.assert_equal(rec.data[3, 5], 1)
        np_test.assert_equal(rec.data.dtype, np.dtype('int'))

        # 2D simpleline thickness 0
        corners = [[0.9, 1.8], [0.9, 4.2], [3.1, 1.8]]
        rec = Parallelogram.make(
            shape=(5, 6), corners=corners, thick=0, dtype='int16')
        np_test.assert_equal((rec.data == 0).sum(), 0)
        np_test.assert_equal((rec.data == -1).sum(), 9)
        np_test.assert_equal((rec.data == 1).sum(), 21)
        np_test.assert_equal(rec.data[3, 4], -1)
        np_test.assert_equal(rec.data[2, 4], -1)
        np_test.assert_equal(rec.data[3, 3], -1)
        np_test.assert_equal(rec.data[2, 3], -1)
        np_test.assert_equal(rec.data[4, 4], 1)
        np_test.assert_equal(rec.data[3, 5], 1)
        np_test.assert_equal(rec.data.dtype, np.dtype('int16'))

        # 2D slanted rectangle
        corners = [[1, 0], [0, 1], [3, 2]]
        rec = Parallelogram.make(shape=(5, 6), corners=corners)
        np_test.assert_equal((rec.data == 0).sum(), 6)
        np_test.assert_equal((rec.data == -1).sum(), 2)
        np_test.assert_equal((rec.data == 1).sum(), 22)
        np_test.assert_equal(rec.data[2, 3], 0)
        np_test.assert_equal(rec.data[1, 2], 0)
        np_test.assert_equal(rec.data[2, 1], 0)
        np_test.assert_equal(rec.data[1, 1], -1)
        np_test.assert_equal(rec.data[2, 2], -1)
        np_test.assert_equal(rec.data[3, 5], 1)

        # 2D parallelogram
        origin = [1, 1]
        basis = [[4, 1], [2, 3]]
        rec = Parallelogram.make(
            shape=(6, 5), origin=origin, basis=basis, thick=0.8)
        desired = [
            [ 1,  1,  1,  1,  1],
            [ 1,  0,  1,  1,  1],
            [ 1,  0, -1,  0,  1],
            [ 1,  0, -1,  0,  1],
            [ 1,  0, -1,  0,  1],
            [ 1,  1,  1,  0,  1]]
        np_test.assert_equal(rec.data, desired)

        # 2D parallelogram, increased thickness
        origin = [1, 1]
        basis = [[4, 1], [2, 3]]
        rec = Parallelogram.make(
            shape=(6, 5), origin=origin, basis=basis, thick=1.2)
        desired = [
            [ 1,  1,  1,  1,  1],
            [ 1,  0,  0,  1,  1],
            [ 1,  0,  0,  0,  1],
            [ 1,  0, -1,  0,  1],
            [ 1,  0,  0,  0,  1],
            [ 1,  1,  0,  0,  1]]
        np_test.assert_equal(rec.data, desired)

        # 2D parallelogram, borderline thick
        origin = [1, 1]
        basis = [[4, 1], [2, 3]]
        rec = Parallelogram.make(
            shape=(6, 5), origin=origin, basis=basis, thick=1)
        desired = [
            [ 1,  1,  1,  1,  1],
            [ 1,  0,  0,  1,  1],
            [ 1,  0,  0,  0,  1],
            [ 1,  0, -1,  0,  1],
            [ 1,  0,  0,  0,  1],
            [ 1,  1,  0,  0,  1]]
        np_test.assert_equal(rec.data, desired)

        # 2D parallelogram, borderline thick
        origin = [1.1, 1]
        basis = [[4.1, 1], [2.1, 3]]
        rec = Parallelogram.make(
            shape=(6, 5), origin=origin, basis=basis, thick=1)
        desired = [
            [ 1,  1,  1,  1,  1],
            [ 1,  0,  1,  1,  1],
            [ 1,  0,  0,  0,  1],
            [ 1,  0, -1,  0,  1],
            [ 1,  0, -1,  0,  1],
            [ 1,  1,  0,  0,  1]]
        np_test.assert_equal(rec.data, desired)

        # 2D parallelogram, different labels
        origin = [1.1, 1]
        basis = [[4.1, 1], [2.1, 3]]
        rec = Parallelogram.make(
            shape=(6, 5), origin=origin, basis=basis, thick=1, surface_label=3,
            inside_label=2, outside_label=4)
        desired = [
            [ 4,  4,  4,  4,  4],
            [ 4,  3,  4,  4,  4],
            [ 4,  3,  3,  3,  4],
            [ 4,  3,  2,  3,  4],
            [ 4,  3,  2,  3,  4],
            [ 4,  4,  3,  3,  4]]
        np_test.assert_equal(rec.data, desired)

        # 2D parallelogram, shifted labels
        origin = [1.1, 1]
        basis = [[4.1, 1], [2.1, 3]]
        rec = Parallelogram.make(
            shape=(6, 5), origin=origin, basis=basis, thick=1, surface_label=1,
            inside_label=0, outside_label=-1)
        desired = [
            [ -1, -1, -1, -1, -1],
            [ -1,  1, -1, -1, -1],
            [ -1,  1,  1,  1, -1],
            [ -1,  1,  0,  1, -1],
            [ -1,  1,  0,  1, -1],
            [ -1, -1,  1,  1, -1]]
        np_test.assert_equal(rec.data, desired)

        # 2D parallelogram, merged labels
        origin = [1.1, 1]
        basis = [[4.1, 1], [2.1, 3]]
        rec = Parallelogram.make(
            shape=(6, 5), origin=origin, basis=basis, thick=1, surface_label=1,
            inside_label=1, outside_label=0)
        desired = [
            [  0,  0,  0,  0,  0],
            [  0,  1,  0,  0,  0],
            [  0,  1,  1,  1,  0],
            [  0,  1,  1,  1,  0],
            [  0,  1,  1,  1,  0],
            [  0,  0,  1,  1,  0]]
        np_test.assert_equal(rec.data, desired)

        # 2D parallelogram, borderline thick
        origin = [1.1, 1]
        basis = [[4.1, 1], [2.1, 3]]
        rec = Parallelogram.make(
            shape=(6, 5), origin=origin, basis=basis, thick=1, surface_label=1,
            inside_label=0, outside_label=1)
        desired = [
            [ 1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1],
            [ 1,  1,  1,  1,  1],
            [ 1,  1,  0,  1,  1],
            [ 1,  1,  0,  1,  1],
            [ 1,  1,  1,  1,  1]]
        np_test.assert_equal(rec.data, desired)

    def test_make_image(self):
        """
        Tests make_image()
        """

        origin = [1, 2, 3]
        basis = [[5, 2, 3], [1, 5, 3], [1, 2, 5]]
        paral = Parallelogram(origin=origin, basis=basis)
        paral.make_image(shape=(8, 8, 8), thick=0, dtype='int16')
        np_test.assert_equal((paral.data[0,:,:] == 1).all(), True)
        np_test.assert_equal((paral.data[1,2:6,3:6] == 0).all(), True)
        np_test.assert_equal((paral.data[5,2:6,3:6] == 0).all(), True)
        np_test.assert_equal((paral.data[6,:,:] == 1).all(), True)
        np_test.assert_equal((paral.data[:,1,:] == 1).all(), True)
        np_test.assert_equal((paral.data[1:6,2,3:6] == 0).all(), True)
        np_test.assert_equal((paral.data[1:6,5,3:6] == 0).all(), True)
        np_test.assert_equal((paral.data[:,6,:] == 1).all(), True)
        np_test.assert_equal((paral.data[:,:,2] == 1).all(), True)
        np_test.assert_equal((paral.data[1:6,2:6,3] == 0).all(), True)
        np_test.assert_equal((paral.data[1:6,2:6,5] == 0).all(), True)
        np_test.assert_equal((paral.data[:,:,6] == 1).all(), True)
        np_test.assert_equal((paral.data[2:5,3:5,4:5] == -1).all(), True)
        np_test.assert_equal(paral.data.dtype, np.dtype('int16'))

    def test_get_all_corners(self):
        """
        Tests get_all_corners()
        """

        # 2D rectangle
        origin = [0, 1]
        basis = [[0, 2], [4, 1]]
        non_basis = [4, 2]
        rec = Parallelogram(origin=origin, basis=basis)
        all = rec.get_all_corners()
        desired = np.vstack((origin, basis, non_basis))
        desired_list = [x.tolist() for x in desired]
        for all_one in all:
            np_test.assert_equal(all_one.tolist() in desired_list, True)

        # 2D rectangle, slanted
        origin = [1, 1]
        basis = [[0, 2], [3, 3]]
        non_basis = [2, 4]
        rec = Parallelogram()
        all = rec.get_all_corners(origin=origin, basis=basis)
        desired = np.vstack((origin, basis, non_basis))
        desired_list = [x.tolist() for x in desired]
        for all_one in all:
            np_test.assert_equal(all_one.tolist() in desired_list, True)

        # 2D parallelogram, slanted
        origin = [1, 1]
        basis = [[0, 2], [5, 2]]
        non_basis = [4, 3]
        rec = Parallelogram(origin=origin, basis=basis)
        all_corners = rec.get_all_corners()
        desired = np.vstack((origin, basis, non_basis))
        np_test.assert_equal(all_corners.shape, desired.shape)
        desired_list = [x.tolist() for x in desired]
        for all_one in all_corners:
            np_test.assert_equal(all_one.tolist() in desired_list, True)

        # 3D rectangle, slanted
        origin = [1, 2, 3]
        basis = [[2, 1, 3], [4, 5, 3], [1, 2, 6]]
        non_basis = [[5, 4, 3], [2, 1, 6], [4, 5, 6], [5, 4, 6]]
        rec = Parallelogram()
        all_corners = rec.get_all_corners(origin=origin, basis=basis)
        desired = np.vstack((origin, basis, non_basis))
        np_test.assert_equal(all_corners.shape, desired.shape)
        desired_list = [x.tolist() for x in desired]
        for all_one in all_corners:
            np_test.assert_equal(all_one.tolist() in desired_list, True)

        # 3D parallelogram, floats
        origin = [3, 2, 1]
        basis = [[4, 4, 1.5], [2, 5, 3.5], [1, 2, 6]]
        non_basis = [[3, 7, 4], [2, 4, 6.5], [0, 5, 8.5], [1, 7, 9]]
        rec = Parallelogram()
        all_corners = rec.get_all_corners(origin=origin, basis=basis)
        desired = np.vstack((origin, basis, non_basis))
        np_test.assert_equal(all_corners.shape, desired.shape)
        for all_one in all_corners:
            compare = np.abs(all_one - desired) < self.precision
            np_test.assert_equal(compare.all(axis=1).any(), True)

        # 3D parallelogram, more floats
        origin = [2.9, 2.2, 1]
        basis = [[3.9, 4.2, 1.5], [1.9, 5.2, 3.5], [0.9, 2.2, 6]]
        non_basis = [
            [2.9, 7.2, 4], [1.9, 4.2, 6.5], [-0.1, 5.2, 8.5], [0.9, 7.2, 9]]
        rec = Parallelogram()
        all_corners = rec.get_all_corners(origin=origin, basis=basis)
        desired = np.vstack((origin, basis, non_basis))
        np_test.assert_equal(all_corners.shape, desired.shape)
        for all_one in all_corners:
            compare = np.abs(all_one - desired) < self.precision
            np_test.assert_equal(compare.all(axis=1).any(), True)

    def test_is_equivalent(self):
        """
        Tests is_equivalent()
        """

        # 2D rectangles
        origin = [0, 1]
        basis = [[0, 2], [4, 1]]
        paral_1 = Parallelogram(origin=origin, basis=basis)
        origin = [4, 2]
        basis = [[4, 1], [0, 2]]
        paral_2 = Parallelogram(origin=origin, basis=basis)
        np_test.assert_equal(paral_1.is_equivalent(paral_2), True)

    def test_get_bounding_box(self):
        """
        Tests get_bounding_box()
        """

        # 2D rectangle, slanted
        origin = [1, 1]
        basis = [[0, 2], [3, 3]]
        #non_basis = [2, 4]
        rec = Parallelogram(origin=origin, basis=basis)
        bb = rec.get_bounding_box(form='min-max')
        np_test.assert_equal(bb, [[0, 3], [1, 4]])

        # 2D parallelogram, slanted
        origin = [1, 1]
        basis = [[0, 2], [5, 2]]
        #non_basis = [4, 3]
        rec = Parallelogram()
        bb = rec.get_bounding_box(origin=origin, basis=basis)
        np_test.assert_equal(bb, [[0, 5], [1, 3]])

        # 2D parallelogram, slanted, slices
        origin = [1, 1]
        basis = [[0, 2], [5, 2]]
        #non_basis = [4, 3]
        rec = Parallelogram()
        bb = rec.get_bounding_box(origin=origin, basis=basis, form='slice')
        np_test.assert_equal(bb, [slice(0, 6), slice(1, 4)])

        # 3D parallelogram
        origin = [3, 2, 1]
        basis = [[4, 4, 1.5], [2, 5, 3.5], [1, 2, 6]]
        #non_basis = [[3, 7, 4], [2, 4, 6.5], [0, 5, 8.5], [1, 7, 9]]
        rec = Parallelogram(origin=origin, basis=basis)
        bb = rec.get_bounding_box(form='min-max_exact')
        np_test.assert_equal(bb, [[0, 4], [2, 7], [1, 9]])

        # 3D parallelogram, more floats, different formats, w/wo extend
        origin = [2.9, 2.2, 1]
        basis = [[3.9, 4.2, 1.5], [1.9, 5.2, 3.5], [0.9, 2.2, 6]]
        #non_basis = [
        #    [2.9, 7.2, 4], [1.9, 4.2, 6.5], [-0.1, 5.2, 8.5], [0.9, 7.2, 9]]
        rec = Parallelogram()
        bb = rec.get_bounding_box(
            origin=origin, basis=basis, form='min-max_exact')
        np_test.assert_almost_equal(bb, [[-0.1, 3.9], [2.2, 7.2], [1, 9]])
        bb = rec.get_bounding_box(
            origin=origin, basis=basis, form='min-max')
        np_test.assert_equal(bb, [[-1, 4], [2, 8], [1, 9]])
        bb = rec.get_bounding_box(
            origin=origin, basis=basis, form='min-len')
        np_test.assert_equal(bb, [[-1, 6], [2, 7], [1, 9]])
        bb = rec.get_bounding_box(
            origin=origin, basis=basis, form='slice')
        np_test.assert_equal(
            [[bb_one.start, bb_one.stop] for bb_one in bb],
            [[-1, 5], [2, 9], [1, 10]])
        bb = rec.get_bounding_box(
            origin=origin, basis=basis, form='min-max_exact',
            extend=[0.2, 1, 1.6])
        np_test.assert_almost_equal(
            bb, [[-0.3, 4.1], [1.2, 8.2], [-0.6, 10.6]])
        bb = rec.get_bounding_box(
            origin=origin, basis=basis, form='slice',
            extend=[0.2, 1, 1.6])
        np_test.assert_equal(
            [[bb_one.start, bb_one.stop] for bb_one in bb],
            [[-1, 6], [1, 10], [-1, 12]])

        # restrict to shape
        origin = [2.9, 2.2, 1]
        basis = [[3.9, 4.2, 1.5], [1.9, 5.2, 3.5], [0.9, 2.2, 6]]
        #non_basis = [
        #    [2.9, 7.2, 4], [1.9, 4.2, 6.5], [-0.1, 5.2, 8.5], [0.9, 7.2, 9]]
        rec = Parallelogram()
        bb = rec.get_bounding_box(
            origin=origin, basis=basis, form='min-max_exact', shape=(5, 6, 7))
        np_test.assert_almost_equal(bb, [[0, 3.9], [2.2, 5], [1, 6]])

        # check that bounding box always has order [min, max]
        origin = [6, 6]
        basis = [[5, 4], [0, 3]]
        rec = Parallelogram()
        bb = rec.get_bounding_box(
            origin=origin, basis=basis, form='min-max_exact')
        np_test.assert_almost_equal(bb, [[-1, 6], [1, 6]])

        # check that bounding box always has order [min, max]
        origin = [5, 6, 7]
        basis = [[2, 5.5, 6], [4.1, 2.2, 5], [3.9, 5, 2.9]]
        rec = Parallelogram()
        bb = rec.get_bounding_box(
            origin=origin, basis=basis, form='min-max_exact')
        for one_ax in bb:
            np_test.assert_equal(one_ax[0] <= one_ax[1], True)

    def test_from_bounding_box(self):
        """
        Test from_bounding_box()
        """

        # no offset
        box = [[1, 11], [2, 22], [4, 44]]
        par = Parallelogram.from_bounding_box(box=box)
        np_test.assert_equal(par.origin, [1, 2, 4])
        desired = [[11, 2, 4], [1, 22, 4], [1, 2, 44]]
        for base_ind in range(par.basis.shape[0]):
            np_test.assert_equal(
                par.basis[base_ind, :].tolist() in desired, True)

        # offset
        box = [[1, 11], [2, 22], [4, 44]]
        par = Parallelogram.from_bounding_box(box=box, offset=1)
        np_test.assert_equal(par.origin, [0, 1, 3])
        desired = [[10, 1, 3], [0, 21, 3], [0, 1, 43]]
        for base_ind in range(par.basis.shape[0]):
            np_test.assert_equal(
                par.basis[base_ind, :].tolist() in desired, True)

    def test_get_previous_bounding_box(self):
        """
        Tests previous_bounding_box()
        """

        # 2D
        box = [[2, 4], [3, 4]]
        phi = np.pi / 2
        center = [2, 3]
        desired_box = [[2, 3], [1, 3]]
        af = Affine2D(phi=phi, d=0, scale=1)
        rec = Parallelogram()
        orig_box = rec.get_previous_bounding_box(
            box=box, tf=af, center=center, form='min-max')
        np_test.assert_equal(
            np.sort(orig_box, axis=1), desired_box)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestParallelogram)
    unittest.TextTestRunner(verbosity=2).run(suite)
