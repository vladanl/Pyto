"""
Tests class BoundarySmooth

# Author: Vladan Lucic
# $Id$
"""
__version__ = "$Revision$"

import os
import unittest

import numpy as np
import numpy.testing as np_test
import scipy as sp

from pyto.spatial.boundary import BoundarySmooth 


class TestBoundarySmooth(np_test.TestCase):
    """
    Tests BoundaryNormal
    """

    def setUp(self):
        """
        """

        unbin_factor = 4
        self.segment_id = 5
        self.external_id_1 = 3
        self.external_id_2 = 1
        slice3 = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0]])
        image_pre = np.stack([slice3, slice3, slice3], axis=2)
        image = self.segment_id * sp.ndimage.zoom(
            image_pre, zoom=unbin_factor, order=0)
        image[:10, :10, :] = np.where(
            image[:10, :10, :] == 0, self.external_id_1, image[:10, :10, :])
        image[10:, 10:, :] = np.where(
            image[10:, 10:, :] == 0, self.external_id_2, image[10:, 10:, :])
        self.image = image
        self.desired_3none = np.array(
            [[3, 3, 3, 0],
             [3, 0, 0, 5],
             [3, 0, 5, 5],
             [3, 5, 5, 5]])
        self.desired_3 = np.array(
            [[3, 3, 3, 0],
             [3, 3, 3, 5],
             [3, 3, 5, 5],
             [3, 5, 5, 5]])
        self.desired_3_inv = np.array(
            [[3, 3, 3, 0],
             [3, 3, 5, 5],
             [3, 5, 5, 5],
             [3, 5, 5, 5]])
        self.desired_1none = np.array(
            [[5, 5, 5, 1],
             [5, 5, 0, 1],
             [5, 0, 0, 1],
             [1, 1, 1, 1]])
        self.desired_1 = np.array(
            [[5, 5, 5, 1],
             [5, 5, 1, 1],
             [5, 1, 1, 1],
             [1, 1, 1, 1]])
        
    def test_morphology_pipe(self):
        """Tests morphology_pipe
        """

        # smooth wo external 
        operations = 'ddeeeedd'
        bs = BoundarySmooth(
            image=self.image, segment_id=self.segment_id, external_id=None)
        im_smooth = bs.morphology_pipe(operations=operations)
        np_test.assert_array_equal(
            im_smooth[10:14, 10:14, 1], self.desired_1none)
        np_test.assert_array_equal(im_smooth[2:6, 7:11, 1], self.desired_3none)

        # smooth wo external 
        operations = 'eeddddee'
        bs = BoundarySmooth(
            image=self.image, segment_id=self.segment_id, external_id=None)
        im_smooth = bs.morphology_pipe(operations=operations)
        np_test.assert_array_equal(
            im_smooth[10:14, 10:14, 1], self.desired_1none)
        np_test.assert_array_equal(im_smooth[2:6, 7:11, 1], self.desired_3none)

        # smooth external_1
        operations = 'ddeeeedd'
        bs = BoundarySmooth(
            image=self.image, segment_id=self.segment_id,
            external_id=self.external_id_1)
        im_smooth = bs.morphology_pipe(operations=operations)
        np_test.assert_array_equal(
            im_smooth[10:14, 10:14, 1], self.desired_1none)
        np_test.assert_array_equal(im_smooth[2:6, 7:11, 1], self.desired_3)

        # smooth external_2
        operations = 'ddeeeedd'
        bs = BoundarySmooth(
            image=self.image, segment_id=self.segment_id,
            external_id=self.external_id_2)
        im_smooth = bs.morphology_pipe(operations=operations)
        np_test.assert_array_equal(im_smooth[10:14, 10:14, 1], self.desired_1)
        np_test.assert_array_equal(im_smooth[2:6, 7:11, 1], self.desired_3none)
        
        # smooth both external
        operations = 'ddeeeedd'
        bs = BoundarySmooth(
            image=self.image, segment_id=self.segment_id,
            external_id=[self.external_id_1, self.external_id_2])
        im_smooth = bs.morphology_pipe(operations=operations)
        np_test.assert_array_equal(im_smooth[10:14, 10:14, 1], self.desired_1)
        np_test.assert_array_equal(im_smooth[2:6, 7:11, 1], self.desired_3)
        desired = np.zeros((3, 3), dtype=int)
        desired[1, 2] = self.segment_id
        desired[2, 1:] = self.segment_id
        np_test.assert_array_equal(im_smooth[:3, 10:13, 1], desired)
        desired = np.zeros((2, 3), dtype=int)
        desired[0, :2] = self.segment_id
        desired[1, 0] = self.segment_id
        np_test.assert_array_equal(im_smooth[8:10, 13:, 1], desired)
        
        # smooth two segments
        operations = 'ddeeeedd'
        bs = BoundarySmooth(
            image=self.image, external_id=self.segment_id,
            segment_id=[self.external_id_1, self.external_id_2])
        im_smooth = bs.morphology_pipe(operations=operations)
        np_test.assert_array_equal(im_smooth[10:14, 10:14, 1], self.desired_1)
        np_test.assert_array_equal(im_smooth[2:6, 7:11, 1], self.desired_3_inv)
        np_test.assert_array_equal(
            im_smooth[:3, 10:13, 1], np.zeros((3, 3), dtype=int))
        np_test.assert_array_equal(
            im_smooth[8:10, 13:, 1], np.zeros((2, 3), dtype=int))
        
        
if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestBoundarySmooth)
    unittest.TextTestRunner(verbosity=2).run(suite)
