"""

Tests module distance

# Author: Vladan Lucic
# $Id$
"""
from __future__ import absolute_import

__version__ = "$Revision$"

from copy import copy, deepcopy
import unittest

import numpy
import numpy.testing as np_test 
import scipy

#from pyto.segmentation.test import common
from pyto.segmentation.distance import Distance
from pyto.segmentation.segment import Segment

class TestDistance(np_test.TestCase):
    """
    """

    def setUp(self):
        """
        """

        ar_1 = numpy.array(
            [[1, 1, 1, 1, 1, 0, 0, 2, 2],
             [0, 0, 0, 0, 0, 0, 0, 2, 2],
             [0, 0, 3, 0, 0, 0, 0, 0, 0],
             [0, 3, 3, 0, 0, 0, 0, 0, 4]])
        self.segments_1 = Segment(data=ar_1) 

    def testCalculate(self):
        """
        Tests calculate() and implicitly getDistance() and setDistance().
        """

        distance = Distance()

        # simple test, mode='min'
        distance.calculate(segments=self.segments_1, ids=(1,2))
        np_test.assert_almost_equal(distance.getDistance(ids=(1,2)), 3)
        np_test.assert_almost_equal(distance.getDistance(ids=(2,1)), 3)
        np_test.assert_almost_equal(
            distance.getDistance(ids=(3,1)) is None, True)
        
        # simple test, mode='max'
        distance.calculate(
            segments=self.segments_1, ids=(1, 2), mode='max', force=True)
        np_test.assert_almost_equal(distance.getDistance(ids=(1, 2)), 7)
        distance.calculate(
            segments=self.segments_1, ids=(2, 1), mode='max', force=True)
        np_test.assert_almost_equal(
            distance.getDistance(ids=(2, 1)), numpy.sqrt(17))

        # another distance
        distance.calculate(segments=self.segments_1, ids=(3,1))
        np_test.assert_almost_equal(distance.getDistance(ids=(1,3)), 2)

        # check arg force
        distance.calculate(segments=self.segments_1, ids=(1,2), force=True)
        self.segments_1.data[0,5] = 1
        np_test.assert_almost_equal(distance.getDistance(ids=(2,1)), 3)
        np_test.assert_almost_equal(
            distance.calculate(segments=self.segments_1, 
                               ids=(1,2), force=False), 3)
        np_test.assert_almost_equal(
            distance.calculate(segments=self.segments_1, ids=(1,2), force=True),
            2)
        np_test.assert_almost_equal(distance.getDistance(ids=(2,1)), 2)
        self.segments_1.data[0,5] = 0

    def test_calculate_symmetric_distance(self):
        """
        Tests calculate_symmetric distance()
        """

        distance = Distance()

        # simple test, mode='min'
        distance.calculate_symmetric_distance(
            segments=self.segments_1, ids=(1,2))
        np_test.assert_almost_equal(distance.getDistance(ids=(1, 2)), 3)

        # mode='max'
        distance.calculate_symmetric_distance(
            segments=self.segments_1, ids=(1, 2), mode='max')
        np_test.assert_almost_equal(distance.getDistance(ids=(1, 2)), 3)
        distance.calculate_symmetric_distance(
            segments=self.segments_1, ids=(1, 2), mode='max', force=True)
        np_test.assert_almost_equal(
            distance.getDistance(ids=(1, 2)), numpy.sqrt(65))

        # mode = 'median'
        distance.calculate_symmetric_distance(
            segments=self.segments_1, ids=(1, 4), mode='median')
        np_test.assert_almost_equal(distance.getDistance(
            ids=(1, 4)), numpy.sqrt(45))
        
       # mode = 'mean'
        distance.calculate_symmetric_distance(
            segments=self.segments_1, ids=(4, 3), mode='mean')
        np_test.assert_almost_equal(distance.getDistance(
            ids=(4, 3)), (6 + 7 + numpy.sqrt(37))/3)
        
if __name__ == '__main__':
    suite = \
        unittest.TestLoader().loadTestsFromTestCase(TestDistance)
    unittest.TextTestRunner(verbosity=2).run(suite)
